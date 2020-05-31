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

        global scores, input_docs, position, learning_rate, sess, train_step_pos, train_step_neg, cross_entropy_pos, \
            cross_entropy_neg, grads_vars, prob, cross_entropy, train_step, doc_length

        input_docs = tf.placeholder(tf.float32, [None, self.Nfeature])
        position = tf.placeholder(tf.int64)
        learning_rate = tf.placeholder(tf.float32, shape=[])
        doc_length = tf.placeholder(tf.int32)

        ########## Liner ranker ########
        W1 = tf.Variable(tf.truncated_normal([self.Nfeature, 1], stddev=0.1 / np.sqrt(float(Nfeature))))
        # b1 = tf.Variable(tf.zeros([1, hidden_units]))
        h1 = tf.matmul(input_docs, W1)
        scores = tf.transpose(h1)
        ##################################

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
        opt = tf.train.GradientDescentOptimizer(learning_rate)
        grads_vars = opt.compute_gradients(cross_entropy)
        train_step = opt.apply_gradients(grads_vars)

        # cross_entropy_pos = tf.reduce_sum(
        #     -tf.log(tf.clip_by_value(prob, 1e-10, 1.0)) * tf.one_hot(position, doc_length),
        #     axis=1)
        # pos_train_opt = tf.train.GradientDescentOptimizer(learning_rate)
        # grads_vars_pos = pos_train_opt.compute_gradients(cross_entropy_pos)
        # train_step_pos = pos_train_opt.apply_gradients(grads_vars_pos)
        # #
        #
        # cross_entropy_neg = tf.reduce_sum(
        #     -tf.log(tf.clip_by_value(1.0 - prob, 1e-10, 1.0)) * tf.one_hot(position, doc_length),
        #     axis=1)
        # neg_train_opt = tf.train.GradientDescentOptimizer(learning_rate)
        # grads_vars_neg = neg_train_opt.compute_gradients(cross_entropy_neg)
        # train_step_neg = neg_train_opt.apply_gradients(grads_vars_neg)

        # Start TF session

        sess = tf.Session()
        sess.run(init)

    # def TF_batch_update(self, dataset):
    #
    #     batch = np.random.randint(len(self.memory), size=self.batch_size)
    #     for i in batch:
    #         query, ranklist, rewards = self.memory[i]
    #         feature_matrix = dataset.get_all_features_by_query(query)
    #         ndoc = len(ranklist)
    #         lenghth = min(self.Lenepisode, ndoc)
    #
    #         for pos in range(lenghth):
    #
    #             loss, _ = sess.run([cross_entropy, train_step],
    #                                feed_dict={input_docs: feature_matrix[ranklist], position: [0],
    #                                           learning_rate: self.lr * rewards[pos]})
    #             ranklist = np.delete(ranklist, 0)

    def TFupdate_policy_trust(self, query, ranklist, rewards, dataset):
        feature_matrix = dataset.get_all_features_by_query(query)
        ndoc = len(ranklist)
        lenghth = min(self.Lenepisode, ndoc)

        for pos in range(lenghth):
            loss, _ = sess.run([cross_entropy, train_step],
                               feed_dict={input_docs: feature_matrix[ranklist], position: [0],
                                          learning_rate: self.lr * rewards[pos]})
            ranklist = np.delete(ranklist, 0)

            # ranklist = np.delete(ranklist, 0)

            # if rewards[pos] < 0:
            #     loss, _ = sess.run([cross_entropy_pos, train_step_pos],
            #                        feed_dict={input_docs: feature_matrix[ranklist],
            #                                   position: [0],
            #                                   doc_length: len(ranklist),
            #                                   learning_rate: self.lr * rewards[pos]})
            # else:
            #     loss, _ = sess.run([cross_entropy_neg, train_step_neg],
            #                        feed_dict={input_docs: feature_matrix[ranklist],
            #                                   position: [0],
            #                                   doc_length: len(ranklist),
            #                                   learning_rate: self.lr * -rewards[pos]})

            # loss, _ = sess.run([cross_entropy_pos, train_step_pos],
            #                    feed_dict={input_docs: feature_matrix[ranklist],
            #                               position: [0],
            #                               doc_length: len(ranklist),
            #                               learning_rate: self.lr * rewards[pos]})

            # ranklist = np.delete(ranklist, 0)

    def TFupdate_reward_trust(self, query, ranklist, rewards, dataset):
        feature_matrix = dataset.get_all_features_by_query(query)
        ndoc = len(ranklist)
        lenghth = min(self.Lenepisode, ndoc)

        for pos in range(lenghth):
            if rewards[pos] > 0:
                loss, _ = sess.run([cross_entropy_pos, train_step_pos],
                                   feed_dict={input_docs: feature_matrix[ranklist],
                                              position: [0],
                                              doc_length: len(ranklist),
                                              learning_rate: self.lr * rewards[pos]})
            else:
                loss, _ = sess.run([cross_entropy_neg, train_step_neg],
                                   feed_dict={input_docs: feature_matrix[ranklist],
                                              position: [0],
                                              doc_length: len(ranklist),
                                              learning_rate: self.lr * -rewards[pos]})

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

    def get_query_result_list(self, dataset, query):
        feature_matrix = dataset.get_all_features_by_query(query)
        docid_list = dataset.get_candidate_docids_by_query(query)
        ndoc = len(docid_list)

        # k = np.minimum(k, ndoc)

        doc_scores = self.get_scores(feature_matrix)

        scoretmp = doc_scores.tolist()

        positions = list(range(ndoc))
        ranklist = np.zeros(ndoc, dtype=np.int32)

        if ndoc == 1:
            ranklist[0] = positions[0]
            return ranklist

        for position in range(ndoc):
            # policy = np.exp((scoretmp - np.max(scoretmp)) / 10)
            # policy = policy / np.sum(policy)

            # softmax does not handle nan
            # probabilities = np.exp(scoretmp) / np.sum(np.exp(scoretmp))
            probabilities = self.softmax(scoretmp)

            # useless, extremely slow
            # probabilities = tf.nn.softmax(scoretmp)
            # probabilities = sess.run(probabilities).ravel()

            # check if policy contains Nan
            # array_sum = np.sum(probabilities)
            # if np.isnan(array_sum):
            #     print(query)
            #     print(scoretmp)
            #     print(probabilities)

            choice = np.random.choice(len(probabilities), 1, p=probabilities)[0]
            ranklist[position] = positions[choice]

            del scoretmp[choice]
            del positions[choice]

        return ranklist

    def softmax(self, x):
        f = np.exp(x - np.max(x))  # shift values
        return f / f.sum(axis=0)

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

        result = sess.run([scores], feed_dict={input_docs: features})[0].reshape([-1])
        # array_sum = np.sum(result)
        # if np.isnan(array_sum):
        #     print(sess.run([scores], feed_dict={input_docs: features}))
        #     print(sess.run([prob], feed_dict={input_docs: features}))
        return result
