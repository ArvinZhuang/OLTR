from ranker.LinearRanker import LinearRanker
import numpy as np


class CMAESLinearRanker(LinearRanker):
    def __init__(self, num_features, learning_rate, step_size, tau, gamma,
                 learning_rate_decay=1, random_initial=True):
        super().__init__(num_features, learning_rate, learning_rate_decay, random_initial)
        self.tau = tau
        self.step_size = step_size
        self.gamma = gamma

    def get_query_result_list(self, dataset, query):
        # listwise ranking with linear model
        self.docid_list = np.array(dataset.get_candidate_docids_by_query(query))
        self.feature_matrix = dataset.get_all_features_by_query(query)

        scores = self.get_scores(self.feature_matrix)
        props = self._softmax_with_tau(scores).reshape(-1)
        props = (props + 1e-5) / (1 + len(props) * 1e-5)

        sample_size = np.minimum(10, len(self.docid_list))

        self.query_result_list = np.random.choice(self.docid_list, sample_size,
                                             replace=False, p=props)
        return self.query_result_list

    def _softmax_with_tau(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp((x - np.max(x)) / self.tau)

        return e_x / e_x.sum(axis=0)

    def sample_unit_vectors(self, num_rankers):
        unit_vectors = np.random.randn(num_rankers, self.num_features)
        vector_norms = np.sum(unit_vectors ** 2, axis=1) ** (1. / 2)
        unit_vectors = unit_vectors / vector_norms[:, None]
        return unit_vectors

    def sample_canditate_rankers(self, unit_vectors):
        # sample unit vectors
        # sample new candidate weights
        new_weights = self.weights + self.step_size * unit_vectors
        return new_weights

    def fitness(self, canditate_rankers, records, dataset):
        result = 0
        current_ranker = self.weights
        all_ranker = np.vstack((current_ranker, canditate_rankers))  # all rankers weights
        #for record in records:
        record = records[len(records) - 1]
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

        SNIPS = self.compute_SNIPS(log_propensity, propensities, click_label)
        np.set_printoptions(suppress=True)
        result -= SNIPS
        return result / len(record)

    def compute_SNIPS(self, log_propensity, propensities, click_label):
        click_label = np.array(click_label).reshape(-1, 1)
        IPS = np.sum((propensities / log_propensity) * click_label, axis=0) / len(click_label)

        S = np.sum((propensities / log_propensity), axis=0) / len(click_label)
        SNIPS = IPS / S

        Var = np.sum((click_label - SNIPS) ** 2 * (propensities / log_propensity) ** 2, axis=0) / np.sum(
            (propensities / log_propensity) ** 2, axis=0)

        return SNIPS + self.gamma * np.sqrt(Var / len(click_label))

    def compute_IPS(self, log_propensity, propensities, click_label):
        click_label = np.array(click_label).reshape(-1, 1)
        IPS = np.sum((propensities / log_propensity) * click_label, axis=0) / len(click_label)
        return IPS

    def softmax(self, x):
        e_x = np.exp(x - np.max(x)) + 1e-6
        return e_x / (e_x.sum(axis=0) + 1e-6)


def get_doc_indexes(result_list, doc_ids):
    doc_ids = np.array(doc_ids)
    #return np.searchsorted(doc_ids,result_list, sorter=range(len(doc_ids)))
    return [np.where(doc_ids==i)[0][0] for i in result_list]
