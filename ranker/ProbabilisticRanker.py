"""
This class is copy from
"""
import numpy as np
from random import sample, random, shuffle
from scipy.linalg import norm
from collections import defaultdict
import copy


def rank(x, ties, reverse=False):
    if not isinstance(x, np.ndarray):
        print(x, type(x))
        x = np.array(x)
        n = 1
    else:
        n = len(x)
    if ties == "first":
        ix = zip(x, reversed(range(n)), range(n))
    elif ties == "last":
        ix = zip(x, range(n), range(n))
    elif ties == "random":
        ix = zip(x, sample(range(n), n), range(n))
    else:
        raise Exception("Unknown method for breaking ties: \"%s\"" % ties)

    ix = sorted(ix, reverse=reverse)

    # ix.sort(reverse=reverse)
    indexes = [i for _, _, i in ix]

    return [i for _, i in sorted(zip(indexes, range(n)))]

def sample_unit_sphere(n):
    """See http://mathoverflow.net/questions/24688/efficiently-sampling-
    points-uniformly-from-the-surface-of-an-n-sphere"""
    v = np.random.randn(n)
    v /= norm(v)
    return v



class ProbabilisticRanker:
    def __init__(self, step_size, learning_rate, num_features):
        self.learning_rate = learning_rate
        self.step_size = step_size
        self.feature_size= num_features

        #random initialize weight
        unit_vector = np.random.randn(self.feature_size)
        unit_vector /= norm(unit_vector)
        self.weights = unit_vector * 0.01

    def init_ranking(self, features):
        scores = np.dot(features, self.weights.T)
        ranks = rank(scores, ties="random", reverse=False)

        ranked_docids = []

        for docid, r in enumerate(ranks):
            ranked_docids.append((r, docid))

        ranked_docids.sort(reverse=True)

        self.docids = [docid for (_, docid) in ranked_docids]

        ranks = np.asarray([i + 1.0 for i in
            sorted(rank(scores, ties="random", reverse=False))])

        max_rank = len(ranks)
        tmp_val = max_rank / pow(ranks, 3)
        self.probs = tmp_val / sum(tmp_val)


    def next(self):
        """produce the next document by random sampling, or
        deterministically"""

        # if there are no more documents
        if len(self.docids) < 1:
            raise Exception("There are no more documents to be selected")

        # if there's only one document
        if len(self.docids) == 1:
            self.probs = np.delete(self.probs, 0)  # should be empty now
            pick = self.docids.pop()  # pop, because it's a list
            return pick

        # sample if there are more documents
        # how to do this efficiently?
        # take cumulative probabilities, then do binary search?
        # if we sort docs and probabilities, we can start search at the
        # beginning. This will be efficient, because we'll look at the most
        # likely docs first.
        cumprobs = np.cumsum(self.probs)
        pick = -1
        rand = random()  # produces a float in range [0.0, 1.0)
        for pos, cp in enumerate(cumprobs):
            if rand < cp:
                pick = self.docids.pop(pos)  # pop, because it's a list
                break

        if (pick == -1):
            print ("Cumprobs:", cumprobs)
            print ("rand", rand)
            raise Exception("Could not select document!")
        # renormalize
        self.probs = np.delete(self.probs, pos)  # delete, it's a numpy array
        self.probs = self.probs / sum(self.probs)
        return pick


    def get_candidate_weight(self):
        """Delta is a change parameter: how much are your weights affected by
        the weight change?"""
        # Some random value from the n-sphere,
        u = sample_unit_sphere(self.feature_size)

        return self.weights + self.step_size * u, u

    def get_new_candidate(self):
        # Get a new candidate whose weights are slightly changed with strength
        # delta.
        w, u = self.get_candidate_weight()

        candidate_ranker = copy.deepcopy(self)
        candidate_ranker.update_weights(w)

        return candidate_ranker, u

    def update_weights(self, w, alpha = None):
        """update weight vector"""
        if alpha is None:
            self.weights = w
        else:
            self.weights = self.weights + alpha * w

    def get_document_probability(self, docid):
        """get probability of producing doc as the next document drawn"""
        pos = self.docids.index(docid)
        return self.probs[pos]

    def document_count(self):
        return len(self.docids)

    def rm_document(self, docid):
        """remove doc from list of available docs and adjust probabilities"""
        # find position of the document

        pos = self.docids.index(docid)

        # delete doc and renormalize
        self.docids.pop(pos)
        self.probs = np.delete(self.probs, pos)
        self.probs = self.probs / sum(self.probs)
    def get_current_weights(self):
        return self.weights

    def get_score(self, features, weights):
        weights = np.array([weights])
        score = np.dot(features, weights.T)[:, 0]
        return score

    def get_all_query_result_list(self, data_processor, weights):
        query_docid_get_feature = data_processor.get_query_docid_get_feature()
        query_get_all_features = data_processor.get_query_get_all_features()
        query_get_docids = data_processor.get_query_get_docids()

        query_result_list = {}

        totalt = 0
        for query in query_get_docids.keys():
            # listwise ranking with linear model
            docid_list = list(query_docid_get_feature[query].keys())

            score_list = self.get_score(query_get_all_features[query], weights)

            # score_list = score_list.tolist()

            docid_score_list = zip(docid_list, score_list)

            # 0.028

            docid_score_list = sorted(docid_score_list, key=lambda x: x[1], reverse=True)

            (docid, socre) = docid_score_list[0]
            query_result_list[query] = [docid]
            for i in range(1, len(docid_list)):
                (docid, socre) = docid_score_list[i]

                query_result_list[query].append(docid)

        return query_result_list

    def probabilistic_multileave(self, rankers, features, length):
        d = defaultdict(list)

        for i, r in enumerate(rankers):
            d[i] = r.init_ranking(features)

        length = min([length] + [r.document_count() for r in rankers])

        # start with empty document list
        l = []
        # random bits indicate which r to use at each rank
        # a = np.asarray([randint(0, len(rankers) - 1) for _ in range(length)])
        a = []
        pool = []

        while len(a) < length:
            if len(pool) == 0:
                pool = list(range(len(rankers)))
                shuffle(pool)
            a.append(pool.pop())


        for next_a in a:
            # flip coin - which r contributes doc (pre-computed in a)
            select = rankers[next_a]
            others = [r for r in rankers if r is not select]
            # draw doc
            pick = select.next()
            l.append(pick)
            for o in others:
                o.rm_document(pick)

        return (np.asarray(l), a)

    def probabilistic_multileave_outcome(self, l, rankers, clicks, features):
        click_ids = np.where(np.asarray(clicks) == 1)[0]

        if not len(click_ids):  # no clicks, will be a tie
            # return [1/float(len(rankers))]*len(rankers)
            # the decision could be made to give each ranker equal credit in a
            # tie so all rankers get rank 1

            # return [1.0 / float(len(rankers))] * len(rankers)
            return [1] * len(rankers)
        for r in rankers:
            r.init_ranking(features)
        p = self.probability_of_list(l, rankers, click_ids)

        creds = self.credits_of_list(p)

        return self.credits_to_outcome(creds)

    def probability_of_list(self, result_list, rankers, clickedDocs):
        '''
        ARGS:
        - result_list: the multileaved list
        - rankers: a list of rankers
        - clickedDocs: the docIds in the result_list which recieved a click

        RETURNS
        -sigmas: list with for each click the list containing the probability
         that the list comes from each ranker
        '''
        tau = 0.3
        n = len(rankers[0].docids)
        sigmoid_total = np.sum(float(n) / (np.arange(n) + 1) ** tau)
        sigmas = np.zeros([len(clickedDocs), len(rankers)])
        for i, r in enumerate(rankers):
            ranks = np.array(self.get_rank(r, result_list))
            for j in range(len(clickedDocs)):
                click = clickedDocs[j]
                sigmas[j, i] = ranks[click] / (sigmoid_total
                                               - np.sum(float(n) /
                                                        (ranks[: click]
                                                         ** tau)))
        for i in range(sigmas.shape[0]):
            sigmas[i, :] = sigmas[i, :] / np.sum(sigmas[i, :])

        return list(sigmas)

    def get_rank(self, ranker, documents):
        '''
        Return the rank of given documents in given ranker.
        Note: rank is not index (rank is index+1)

        ARGS:
        - ranker
        - documents

        RETURN:
        - a list containing the rank in the ranker for each of the documents
        '''
        ranks = [None] * len(documents)
        docsInRanker = ranker.docids

        for i, d in enumerate(documents):
            if d in docsInRanker:
                ranks[i] = docsInRanker.index(d) + 1
        return ranks

    def credits_of_list(self, p):
        '''
        ARGS:
        -p: list with for each click the list containing the probability that
            the list comes from each ranker

        RETURNS:
        - credits: list of credits for each ranker
        '''
        creds = [np.average(col) for col in zip(*p)]
        return creds

    def credits_to_outcome(self, creds):
        rankers_credits = sorted(zip(range(len(creds)), creds), reverse=True,
                                 key=lambda item: item[1])

        ranked_credits = len(rankers_credits) * [None]
        last_c = None
        last_rank = 0
        rank = 0
        for (r, c) in rankers_credits:
            rank += 1
            if not (c == last_c):
                ranked_credits[r] = rank
                last_rank = rank
            else:
                ranked_credits[r] = last_rank
            last_c = c

        return ranked_credits