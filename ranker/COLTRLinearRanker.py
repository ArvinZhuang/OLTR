from ranker.LinearRanker import LinearRanker
import numpy as np


class COLTRLinearRanker(LinearRanker):
    def __init__(self, num_features, learning_rate, step_size, tau, gamma,
                 learning_rate_decay=1, random_initial=True):
        super().__init__(num_features, learning_rate, learning_rate_decay, random_initial=random_initial)
        self.tau = tau
        self.step_size = step_size
        self.gamma = gamma

    def get_query_result_list(self, dataset, query):
        # listwise ranking with linear model
        self.docid_list = np.array(dataset.get_candidate_docids_by_query(query))
        self.feature_matrix = dataset.get_all_features_by_query(query)

        scores = self.get_scores(self.feature_matrix)
        probs = self._softmax_with_tau(scores).reshape(-1)

        sample_size = np.minimum(10, len(self.docid_list))

        if np.sum(probs > 0) < sample_size:
            safe_size = np.sum(probs > 0)
            query_result_list = np.random.choice(self.docid_list, safe_size, replace=False, p=probs)
            rest = np.setdiff1d(self.docid_list, query_result_list)
            np.random.shuffle(rest)
            self.query_result_list = np.append(query_result_list, rest)
            return query_result_list[:sample_size]

        self.query_result_list = np.random.choice(self.docid_list, sample_size,
                                             replace=False, p=probs)
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

    def infer_winners_renomalize(self, canditate_rankers, record):
        current_ranker = self.weights
        all_ranker = np.vstack((current_ranker, canditate_rankers))  # all rankers weights
        query = record[0]
        result_list = record[1]
        click_label = record[2]
        log_weight = np.array(record[3])

        doc_indexes = [np.where(self.docid_list==i)[0][0] for i in result_list]
        scores = np.dot(self.feature_matrix, all_ranker.T)
        log_score = np.dot(self.feature_matrix, log_weight.T)


        probs = self.softmax(scores)
        log_probs = self.softmax(log_score)
        log_probs = log_probs.reshape(-1, 1)

        propensities = probs[doc_indexes[0]]
        log_propensity = log_probs[doc_indexes[0]]

        #renormalize
        scores[doc_indexes[0]] = np.amin(scores)
        log_score[doc_indexes[0]] = np.amin(log_score)

        scores -= np.amax(scores)
        log_score -= np.amax(log_score)

        exp_scores = np.exp(scores)
        exp_log_score = np.exp(log_score)

        exp_scores[doc_indexes[0]] = 0
        exp_log_score[doc_indexes[0]] = 0

        probs = exp_scores/np.sum(exp_scores)
        log_probs = exp_log_score / np.sum(exp_log_score)
        log_probs = log_probs.reshape(-1, 1)

        for i in range(1, len(doc_indexes)):
            propensities = np.vstack((propensities, probs[doc_indexes[i]]))
            log_propensity = np.vstack((log_propensity, log_probs[doc_indexes[i]]))
            scores[doc_indexes[:i+1]] = np.amin(scores)
            log_score[doc_indexes[:i+1]] = np.amin(log_score)
            scores -= np.amax(scores)
            log_score -= np.amax(log_score)
            exp_scores = np.exp(scores)
            exp_log_score = np.exp(log_score)
            exp_scores[doc_indexes[:i+1]] = 0
            exp_log_score[doc_indexes[:i+1]] = 0
            probs = exp_scores / np.sum(exp_scores+ 1e-8)
            log_probs = exp_log_score / (np.sum(exp_log_score) + 1e-8)
            log_probs = log_probs.reshape(-1, 1)



        SNIPS = self.compute_SNIPS(log_propensity, propensities, click_label)
        winners = np.where(SNIPS < SNIPS[0])[0]
        #
        # IPS = self.compute_IPS(log_propensity, propensities, click_label)
        # winners = np.where(IPS < IPS[0])[0]

        if len(winners) == 0:
            return None
        return winners


    def infer_winners(self, canditate_rankers, record):
        current_ranker = self.weights
        all_ranker = np.vstack((current_ranker, canditate_rankers))  # all rankers weights
        query = record[0]
        result_list = record[1]
        click_label = record[2]
        log_weight = np.array(record[3])

        doc_indexes = [np.where(self.docid_list==i)[0][0] for i in result_list]

        scores = np.dot(self.feature_matrix, all_ranker.T)
        log_score = np.dot(self.feature_matrix, log_weight.T)


        propensities = self.softmax(scores)[doc_indexes]
        log_propensity = self.softmax(log_score)[doc_indexes]
        log_propensity = log_propensity.reshape(len(result_list), 1)

        SNIPS = self.compute_SNIPS(log_propensity, propensities, click_label)

        winners = np.where(SNIPS < SNIPS[0])[0]

        #
        # IPS = self.compute_IPS(log_propensity, propensities, click_label)
        # winners = np.where(IPS < IPS[0])[0]

        if len(winners) == 0:
            return None
        return winners

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
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

