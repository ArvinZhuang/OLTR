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

    def get_SNIPS(self, canditate_rankers, record):
        current_ranker = self.weights
        all_ranker = np.vstack((current_ranker, canditate_rankers))  # all rankers weights
        query = record[0]
        result_list = record[1]
        click_label = record[2]
        log_weight = np.array(record[3])

        doc_indexes = [np.where(self.docid_list == i)[0][0] for i in result_list]

        scores = np.dot(self.feature_matrix, all_ranker.T)
        log_score = np.dot(self.feature_matrix, log_weight.T)

        propensities = self.softmax(scores)[doc_indexes]
        log_propensity = self.softmax(log_score)[doc_indexes]
        log_propensity = log_propensity.reshape(len(result_list), 1)


        SNIPS = self.compute_SNIPS(log_propensity, propensities, click_label)

        winners = np.where(SNIPS < SNIPS[0])[0]

        # IPS = self.compute_IPS(log_propensity, propensities, click_label)
        # winners = np.where(IPS < IPS[0])[0]

        if len(winners) == 0:
            return None
        return SNIPS * -1
