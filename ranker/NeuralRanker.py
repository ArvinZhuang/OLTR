import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from ranker.AbstractRanker import AbstractRanker
import numpy as np

class NeuralRanker(AbstractRanker):
    def __init__(self, num_features, learning_rate):
        super().__init__(num_features)

        self.learning_rate = learning_rate

        self.xs = tf.placeholder(tf.float32, [None, self.num_features])
        self.ys = tf.placeholder(tf.float32, [None, 2])
        self.l1 = self._add_layer(self.xs, self.num_features, 12, activation_function=tf.nn.relu)
        self.l2 = self._add_layer(self.l1, 12, 4, activation_function=tf.nn.relu)
        self.prediction = self._add_layer(self.l2, 4, 2, activation_function=tf.nn.softmax)
        self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.ys * tf.log(self.prediction + 1e-10),
                                                           reduction_indices=[1]))
        self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cross_entropy)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _add_layer(self, inputs, in_size, out_size, activation_function=None, ):
        # add one more layer and return the output of this layer
        Weights = tf.Variable(tf.random_normal([in_size, out_size]))
        biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, )
        Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b, )
        return outputs



    def update(self, clicks, result_list, all_features):
        last_click, click_bitmap = self._clicks_to_bitmap(clicks)
        features = all_features[result_list]

        self.sess.run([self.train_step, self.cross_entropy],
                      feed_dict={self.xs: features[:last_click + 1],
                                 self.ys: click_bitmap})

    def _clicks_to_bitmap(self, clicks):
        last_click = np.where(clicks)[0][-1]
        click_label = clicks[:last_click+1]
        click_label_flip = 1 - clicks[:last_click+1]
        click_bitmap = np.vstack((click_label, click_label_flip)).T
        return last_click, click_bitmap



    def get_query_result_list(self, dataset, query):
        feature_matrix = dataset.get_all_features_by_query(query)
        docid_list = np.array(dataset.get_candidate_docids_by_query(query))
        n_docs = docid_list.shape[0]

        k = np.minimum(10, n_docs)

        doc_scores = self.get_scores(feature_matrix)

        # doc_scores += 18 - np.amax(doc_scores)

        ranking = self._recursive_choice(np.copy(doc_scores),
                                         np.array([], dtype=np.int32),
                                         k)
        return ranking, doc_scores

    def _recursive_choice(self, scores, incomplete_ranking, k_left):
        n_docs = scores.shape[0]

        scores[incomplete_ranking] = np.amin(scores)

        # scores += 18 - np.amax(scores)

        exp_scores = np.exp(scores)


        exp_scores[incomplete_ranking] = 0
        probs = exp_scores / np.sum(exp_scores)

        safe_n = np.sum(probs > 10 ** (-4) / n_docs)

        safe_k = np.minimum(safe_n, k_left)

        next_ranking = np.random.choice(np.arange(n_docs),
                                        replace=False,
                                        p=probs,
                                        size=safe_k)

        ranking = np.concatenate((incomplete_ranking, next_ranking))
        k_left = k_left - safe_k

        if k_left > 0:
            return self._recursive_choice(scores, ranking, k_left)
        else:
            return ranking

    def get_all_query_result_list(self, dataset):
        query_result_list = {}

        for query in dataset.get_all_querys():
            docid_list = np.array(dataset.get_candidate_docids_by_query(query))
            docid_list = docid_list.reshape((len(docid_list), 1))
            feature_matrix = dataset.get_all_features_by_query(query)
            score_list = self.get_scores(feature_matrix)

            docid_score_list = np.column_stack((docid_list, score_list))
            docid_score_list = np.flip(docid_score_list[docid_score_list[:, 1].argsort()], 0)

            query_result_list[query] = docid_score_list[:, 0]

        return query_result_list

    def get_scores(self, features):
        predicts = self.sess.run(self.prediction, feed_dict={self.xs: features})
        return predicts[:,0]

    def assign_weights(self, weights):
        pass

    def get_current_weights(self):
        pass