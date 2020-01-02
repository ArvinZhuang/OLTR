from ranker.COLTRLinearRanker import COLTRLinearRanker
import numpy as np
from collections import defaultdict
from random import sample, random, shuffle
import copy


class NESLinearRanker(COLTRLinearRanker):
    def __init__(self, num_features, learning_rate, mu, cov, sigma, tau, gamma,
                 learning_rate_decay=1, random_initial=True):
        super().__init__(num_features, learning_rate, 1, tau, gamma, learning_rate_decay, random_initial)
        self.sigma = sigma
        self.mu = mu
        self.cov = cov
        self.weights = self.sample_new_pop(1)[0]

    def update(self, dmu, dcov):
        self.mu += self.learning_rate * dmu
        self.cov += self.learning_rate * dcov
        self.learning_rate *= self.learning_rate_decay

    def sample_random_vectors(self, n):
        random_vectors = np.random.randn(n, self.num_features) * self.sigma
        return random_vectors

    def sample_canditate_rankers(self, unit_vectors):
        new_weights = self.weights + unit_vectors
        return new_weights

    def sample_new_pop(self, n):
        weights = np.random.multivariate_normal(self.mu, self.cov, n)
        return weights

    def get_SNIPS(self, canditate_rankers, records, dataset):
        #current_ranker = self.weights
        #all_ranker = np.vstack((current_ranker, canditate_rankers))  # all rankers weights
        all_ranker = canditate_rankers
        select_size = 1
        if (len(records) < select_size) :
            selected = records
        else:
            selected = records[-select_size:]
        for record in selected:
            query = record[0]
            result_list = record[1]
            click_label = record[2]
            log_weight = np.array(record[3])

            doc_indexes = get_doc_indexes(result_list, dataset.get_candidate_docids_by_query(query))
            feature_matrix = dataset.get_all_features_by_query(query)

            scores = np.dot(feature_matrix, all_ranker.T)
            log_score = np.dot(feature_matrix, log_weight.T)

            propensities = self.softmax(scores)[doc_indexes]
            log_propensity = self.softmax(log_score)[doc_indexes]
            log_propensity = log_propensity.reshape(len(result_list), 1)

            try:
                SNIPS += self.compute_SNIPS(log_propensity, propensities, click_label)
            except NameError:
                SNIPS = self.compute_SNIPS(log_propensity, propensities, click_label)
        SNIPS /= len(records)
        winners = np.where(SNIPS < SNIPS[0])[0]

        # IPS = self.compute_IPS(log_propensity, propensities, click_label)
        # winners = np.where(IPS < IPS[0])[0]

        if len(winners) == 0:
            return None
        return SNIPS * -1

    def softmax(self, x):
        e_x = np.exp(x - np.max(x)) + 1e-6
        return e_x / (e_x.sum(axis=0) + 1e-6)

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
            print("Cumprobs:", cumprobs)
            print("rand", rand)
            raise Exception("Could not select document!")
        # renormalize
        self.probs = np.delete(self.probs, pos)  # delete, it's a numpy array
        self.probs = self.probs / sum(self.probs)
        return pick

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
        p = probability_of_list(l, rankers, click_ids)

        creds = credits_of_list(p)

        return creds


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


def probability_of_list(result_list, rankers, clickedDocs):
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
        ranks = np.array(get_rank(r, result_list))
        for j in range(len(clickedDocs)):
            click = clickedDocs[j]
            sigmas[j, i] = ranks[click] / (sigmoid_total
                                           - np.sum(float(n) /
                                                    (ranks[: click]
                                                     ** tau)))
    for i in range(sigmas.shape[0]):
        sigmas[i, :] = sigmas[i, :] / np.sum(sigmas[i, :])

    return list(sigmas)

def get_rank(ranker, documents):
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

def credits_to_outcome(creds):
    rankers_credits = sorted(zip(range(len(creds)), creds), reverse=True,
                             key=lambda item: item[1])


    ranked_credits = len(rankers_credits)*[None]
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

def credits_of_list(p):
    '''
    ARGS:
    -p: list with for each click the list containing the probability that
        the list comes from each ranker
    RETURNS:
    - credits: list of credits for each ranker
    '''
    creds = [np.average(col) for col in zip(*p)]
    return creds

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


def get_doc_indexes(result_list, doc_ids):
    doc_ids = np.array(doc_ids)
    #return np.searchsorted(doc_ids,result_list, sorter=range(len(doc_ids)))
    return [np.where(doc_ids==i)[0][0] for i in result_list]

