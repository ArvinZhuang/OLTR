from ranker.COLTRLinearRanker import COLTRLinearRanker
import numpy as np


class ESLinearRanker(COLTRLinearRanker):
    def __init__(self, num_features, learning_rate, sigma, tau, gamma,
                 learning_rate_decay=1, random_initial=True):
        super().__init__(num_features, learning_rate, 1, tau, gamma, learning_rate_decay, random_initial)
        self.sigma = sigma

    def sample_random_vectors(self, n):
        random_vectors = np.random.randn(n, self.num_features) * self.sigma
        return random_vectors

    def sample_canditate_rankers(self, unit_vectors):
        new_weights = self.weights + unit_vectors
        return new_weights
    def get_nomalized_SNIPS(self, canditate_rankers, record):
        current_ranker = self.weights
        all_ranker = np.vstack((current_ranker, canditate_rankers))  # all rankers weights
        query = record[0]
        result_list = record[1]
        click_label = record[2]
        log_weight = np.array(record[3])

        doc_indexes = [np.where(self.docid_list == i)[0][0] for i in result_list]
        scores = np.dot(self.feature_matrix, all_ranker.T)
        log_score = np.dot(self.feature_matrix, log_weight.T)

        probs = self.softmax(scores)
        log_probs = self.softmax(log_score)
        log_probs = log_probs.reshape(-1, 1)

        propensities = probs[doc_indexes[0]]
        log_propensity = log_probs[doc_indexes[0]]

        # renormalize
        scores[doc_indexes[0]] = np.amin(scores)
        log_score[doc_indexes[0]] = np.amin(log_score)

        scores -= np.amax(scores)
        log_score -= np.amax(log_score)

        exp_scores = np.exp(scores)
        exp_log_score = np.exp(log_score)

        exp_scores[doc_indexes[0]] = 0
        exp_log_score[doc_indexes[0]] = 0

        probs = exp_scores / np.sum(exp_scores)
        log_probs = exp_log_score / np.sum(exp_log_score)
        log_probs = log_probs.reshape(-1, 1)

        for i in range(1, len(doc_indexes)):
            propensities = np.vstack((propensities, probs[doc_indexes[i]]))
            log_propensity = np.vstack((log_propensity, log_probs[doc_indexes[i]]))
            scores[doc_indexes[:i + 1]] = np.amin(scores)
            log_score[doc_indexes[:i + 1]] = np.amin(log_score)
            scores -= np.amax(scores)
            log_score -= np.amax(log_score)
            exp_scores = np.exp(scores)
            exp_log_score = np.exp(log_score)
            exp_scores[doc_indexes[:i + 1]] = 0
            exp_log_score[doc_indexes[:i + 1]] = 0
            probs = exp_scores / np.sum(exp_scores + 1e-8)
            log_probs = exp_log_score / (np.sum(exp_log_score) + 1e-8)
            log_probs = log_probs.reshape(-1, 1)

        SNIPS = self.compute_SNIPS(log_propensity, propensities, click_label)
        winners = np.where(SNIPS < SNIPS[0])[0]
        #
        # IPS = self.compute_IPS(log_propensity, propensities, click_label)
        # winners = np.where(IPS < IPS[0])[0]

        if len(winners) == 0:
            return None
        return SNIPS * -1



    def get_SNIPS(self, canditate_rankers, records, dataset):
        current_ranker = self.weights
        all_ranker = np.vstack((current_ranker, canditate_rankers))  # all rankers weights
        select_size = 50
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
        #print(SNIPS)
        winners = np.where(SNIPS < SNIPS[0])[0]

        # IPS = self.compute_IPS(log_propensity, propensities, click_label)
        # winners = np.where(IPS < IPS[0])[0]

        if len(winners) == 0:
            return None
        return SNIPS * -1

    def softmax(self, x):
        e_x = np.exp(x - np.max(x)) + 1e-6
        return e_x / (e_x.sum(axis=0) + 1e-6)

def get_doc_indexes(result_list, doc_ids):
    doc_ids = np.array(doc_ids)
    #return np.searchsorted(doc_ids,result_list, sorter=range(len(doc_ids)))
    return [np.where(doc_ids==i)[0][0] for i in result_list]
