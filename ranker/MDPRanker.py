from ranker.AbstractRanker import AbstractRanker
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class MDPRanker(AbstractRanker):
    def __init__(self,
                 Nhidden_unit,
                 Nfeature,
                 Learningrate,
                 Lenepisode=10,
                 memory_size=100,
                 batch_size=1):
        super().__init__(Nfeature)
        self.Nfeature = Nfeature
        self.Lenepisode = Lenepisode
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = []
        self.W = np.random.rand(Nfeature)
        # self.W = np.zeros(Nfeature)
        self.lr = Learningrate

        self.Ntop = 10
        self.memory = []
        self.ite = 0

        self.Nhidden_unit = Nhidden_unit

        global scores, input_docs, position, learning_rate, sess, train_step, cross_entropy, grads_vars, prob

        input_docs = tf.placeholder(tf.float32, [None, self.Nfeature])
        position = tf.placeholder(tf.int64)
        learning_rate = tf.placeholder(tf.float32, shape=[])

        # Liner ranker
        W1 = tf.Variable(tf.truncated_normal([self.Nfeature, 1], stddev=0.1 / np.sqrt(float(Nfeature))))
        # b1 = tf.Variable(tf.zeros([1, hidden_units]))
        h1 = tf.matmul(input_docs, W1)
        scores = tf.transpose(h1)

        ########## neural ranker ########
        # Generate hidden layer
        # W1 = tf.Variable(tf.truncated_normal([self.Nfeature, self.Nhidden_unit], stddev=0.1 / np.sqrt(float(Nfeature))))
        # # b1 = tf.Variable(tf.zeros([1, hidden_units]))
        # h1 = tf.tanh(tf.matmul(input_docs, W1))
        #
        # # Second layer -- linear classifier for action logits
        # W2 = tf.Variable(tf.truncated_normal([self.Nhidden_unit, 1], stddev=0.1 / np.sqrt(float(self.Nhidden_unit))))
        # # b2 = tf.Variable(tf.zeros([1]))
        # scores = tf.transpose(tf.matmul(h1, W2))  # + b2
        ##################################
        prob = tf.nn.softmax(scores)

        init = tf.global_variables_initializer()
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, labels=position)
        # train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
        opt = tf.train.GradientDescentOptimizer(learning_rate)
        grads_vars = opt.compute_gradients(cross_entropy)
        train_step = opt.apply_gradients(grads_vars)

        # neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, labels=position)
        # # train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
        # # opt = tf.train.GradientDescentOptimizer(learning_rate)
        # # grads_vars = opt.compute_gradients(cross_entropy)
        # # train_step = opt.apply_gradients(grads_vars)
        # loss = tf.reduce_mean(neg_log_prob * position)
        # self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

        # Start TF session
        sess = tf.Session()
        sess.run(init)

    def TF_batch_update(self, dataset):

        batch = np.random.randint(len(self.memory), size=self.batch_size)
        for i in batch:
            query, ranklist, rewards = self.memory[i]
            feature_matrix = dataset.get_all_features_by_query(query)
            ndoc = len(ranklist)
            lenghth = min(self.Lenepisode, ndoc)

            for pos in range(lenghth):

                loss, _ = sess.run([cross_entropy, train_step],
                                   feed_dict={input_docs: feature_matrix[ranklist], position: [0],
                                              learning_rate: self.lr * rewards[pos]})
                ranklist = np.delete(ranklist, 0)

    def TFupdate(self, query, ranklist, rewards, dataset):
        feature_matrix = dataset.get_all_features_by_query(query)
        ndoc = len(ranklist)
        lenghth = min(self.Lenepisode, ndoc)

        for pos in range(lenghth):

            loss, _ = sess.run([cross_entropy, train_step],
                               feed_dict={input_docs: feature_matrix[ranklist], position: [0],
                                          learning_rate: self.lr * rewards[pos]})
            ranklist = np.delete(ranklist, 0)

    def record_episode(self, query, ranklist, rewards):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        if self.memory_size > len(self.memory):
            self.memory.append([query, ranklist, rewards])
        else:
            index = self.memory_counter % self.memory_size

            self.memory[index] = [query, ranklist, rewards]
        self.memory_counter += 1


    def get_query_result_list(self, dataset, query, k=10):
        feature_matrix = dataset.get_all_features_by_query(query)
        docid_list = dataset.get_candidate_docids_by_query(query)
        ndoc = len(docid_list)

        k = np.minimum(k, ndoc)



        doc_scores = self.get_scores(feature_matrix)

        scoretmp = doc_scores.tolist()

        positions = list(range(ndoc))
        ranklist = np.zeros(k, dtype=np.int32)

        if k == 1:
            ranklist[0] = positions[0]
            return ranklist

        for position in range(k):
            # policy = np.exp((scoretmp - np.max(scoretmp)) / 10)
            # policy = policy / np.sum(policy)
            policy = np.exp(scoretmp) / np.sum(np.exp(scoretmp))
            choice = np.random.choice(len(policy), 1, p=policy)[0]
            ranklist[position] = positions[choice]

            del scoretmp[choice]
            del positions[choice]

        return ranklist

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
        return sess.run([scores], feed_dict={input_docs: features})[0].reshape([-1])