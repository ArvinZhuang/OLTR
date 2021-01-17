from ranker.AbstractRanker import AbstractRanker
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


class MDPRankerV2(AbstractRanker):
    def __init__(self,
                 Nhidden_unit,
                 Nfeature,
                 Learningrate,
                 Lenepisode=10,
                 memory_size=100,
                 batch_size=1,
                 lr_decay=False,
                 loss_type='pointwise'):

        super().__init__(Nfeature)
        tf.reset_default_graph()  # used for multiprocessor training, otherwise has errors

        self.Nfeature = Nfeature
        self.Lenepisode = Lenepisode
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = []
        self.W = np.random.rand(Nfeature)
        # self.W = np.zeros(Nfeature)
        self.lr = Learningrate
        self.loss_type = loss_type
        self.Ntop = 10
        self.memory = []
        self.ite = 0

        self.Nhidden_unit = Nhidden_unit

        self.input_docs = tf.placeholder(tf.float32, [None, self.Nfeature])
        self.position = tf.placeholder(tf.int64)
        self.doc_length = tf.placeholder(tf.int32)
        self.advantage = tf.placeholder(tf.float32)

        self.aW1 = tf.Variable(tf.truncated_normal([self.Nfeature, 1], stddev=0.1 / np.sqrt(float(self.Nfeature))))
        # b1 = tf.Variable(tf.zeros([1, hidden_units]))
        ah1 = tf.matmul(self.input_docs, self.aW1)
        self.doc_scores = tf.transpose(ah1)

        if loss_type == 'pointwise':
            self.prob = tf.nn.softmax(self.doc_scores)

            neg_log_prob = tf.reduce_sum(
                -tf.log(tf.clip_by_value(self.prob, 1e-10, 1.0)) * tf.one_hot(self.position, self.doc_length),
                axis=1)
            self.loss = tf.reduce_mean(neg_log_prob * self.advantage)

        if loss_type == 'pairwise':
            self.position2 = tf.placeholder(tf.int64)
            self.exp = tf.math.exp(tf.clip_by_value(self.doc_scores[0][self.position] - self.doc_scores[0][self.position2], -100, 70))
            self.P = tf.math.divide(self.exp, (1 + self.exp))


            neg_log_prob = tf.reduce_sum(
                -tf.log(tf.clip_by_value(self.P, 1e-10, 1.0)) * tf.one_hot([0], self.doc_length), axis=1)

            self.loss = tf.reduce_mean(neg_log_prob * self.advantage)

        step = tf.Variable(0, trainable=False)

        if lr_decay:
            rate = tf.train.exponential_decay(self.lr, step, 1000, 0.95)
        else:
            rate = self.lr

        # self.train_op = tf.train.AdamOptimizer(rate)
        self.train_op = tf.train.GradientDescentOptimizer(rate)

        # train with gradients accumulative style
        tvs = tf.trainable_variables()
        accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs]
        self.zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]
        self.gvs = self.train_op.compute_gradients(self.loss, tvs)
        self.accum_ops = [accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(self.gvs)]
        self.actor_train_step = self.train_op.apply_gradients([(accum_vars[i], gv[1]) for i, gv in enumerate(self.gvs)])  # gv[1] is equal to the current tvs

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

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

    def update_policy(self, query, ranklist, rewards, dataset):
        feature_matrix = dataset.get_all_features_by_query(query)
        ndoc = len(ranklist)
        lenghth = min(self.Lenepisode, ndoc)

        # print(self.sess.run([self.prob], feed_dict={self.input_docs: feature_matrix[ranklist[:10]]}))

        gradient_vectors = np.zeros((self.num_features, lenghth))
        self.sess.run(self.zero_ops)

        if self.loss_type == "pointwise":
            for pos in range(lenghth):


                self.sess.run([self.accum_ops], feed_dict={self.input_docs: feature_matrix[ranklist[pos:]],
                                                           self.position: [0],
                                                           self.doc_length: len(ranklist[pos:]),
                                                           self.advantage: rewards[pos]})

                gradient = self.sess.run([self.gvs], feed_dict={self.input_docs: feature_matrix[ranklist[pos:]],
                                                           self.position: [0],
                                                           self.doc_length: len(ranklist[pos:]),
                                                           self.advantage: rewards[pos]})[0][0][0]  # get gradient as np.array
                gradient_vectors[:, pos] = gradient.reshape(-1)

            gradient_var = np.sum(np.var(gradient_vectors, axis=1))
            self.sess.run([self.actor_train_step])
            return gradient_var

        # if self.loss_type == "pairwise":
        #
        #     for pos in range(lenghth):
        #         for next_pos in range(1, lenghth-pos):
        #             _, loss = self.sess.run([self.accum_ops, self.loss], feed_dict={self.input_docs: feature_matrix[ranklist[pos:]],
        #                                                        self.position: 0,
        #                                                        self.position2: next_pos,
        #                                                        self.doc_length: len(ranklist[pos:]),
        #                                                        self.advantage: rewards[pos]-rewards[pos+next_pos]})
        #             # array_sum = np.sum(_)
        #             # if np.isnan(array_sum):
        #             #     print(feature_matrix)
        #             #     print("gradient:", _)
        #             #     print("loss:", loss)
        #             #     # x = self.sess.run(self.aW1)
        #             #     # print(x, x.shape)
        #             #     scores = self.sess.run(self.doc_scores, feed_dict={
        #             #         self.input_docs: feature_matrix[ranklist[pos:]]})
        #             #     print(scores)
        #             #     print(next_pos)
        #             #
        #             #     print(np.divide(np.exp(scores[0][0] - scores[0][next_pos]),
        #             #     (1 + np.exp(scores[0][0] - scores[0][next_pos]))))
        #             #
        #             #     exp = self.sess.run(self.exps, feed_dict={
        #             #         self.input_docs: feature_matrix[ranklist[pos:]],
        #             #         self.position: 0,
        #             #         self.position2: next_pos})
        #             #     print(exp)
        #             #     raise Exception
        #
        #     self.sess.run([self.actor_train_step])



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

        result = self.sess.run([self.doc_scores], feed_dict={self.input_docs: features})[0].reshape([-1])
        # array_sum = np.sum(result)
        # if np.isnan(array_sum):
        #     print(features)
        #     x = self.sess.run(self.aW1)
        #     print(x, x.shape)
        #     print(self.sess.run([self.doc_scores], feed_dict={self.input_docs: features}))
            # print(self.sess.run([self.prob], feed_dict={self.input_docs: features}))
        return result
