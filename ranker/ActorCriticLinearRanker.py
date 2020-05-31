from ranker.AbstractRanker import AbstractRanker
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


class ActorCriticLinearRanker(AbstractRanker):
    def __init__(self,
                 Nfeature,
                 Learningrate,
                 Nhidden_unit=256,
                 Lenepisode=10,
                 memory_size=100,
                 batch_size=1):
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

        self.Ntop = 10
        self.memory = []
        self.ite = 0

        self.Nhidden_unit = Nhidden_unit



        self._build_actor()
        self._build_critic()
        # Start TF session
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _build_actor(self):
        self.input_docs = tf.placeholder(tf.float32, [None, self.Nfeature])
        self.position = tf.placeholder(tf.int64)
        self.doc_length = tf.placeholder(tf.int32)
        self.advantage = tf.placeholder(tf.float32)

        aW1 = tf.Variable(tf.truncated_normal([self.Nfeature, 1], stddev=0.1 / np.sqrt(float(self.Nfeature))))
        # b1 = tf.Variable(tf.zeros([1, hidden_units]))
        ah1 = tf.matmul(self.input_docs, aW1)
        self.doc_scores = tf.transpose(ah1)
        self.prob = tf.nn.softmax(self.doc_scores)

        neg_log_prob = tf.reduce_sum(
            -tf.log(tf.clip_by_value(self.prob, 1e-10, 1.0)) * tf.one_hot(self.position, self.doc_length),
            axis=1)
        loss = tf.reduce_mean(neg_log_prob * self.advantage)

        self.train_op = tf.train.AdamOptimizer(self.lr)

        # train with gradients accumulative style
        tvs = tf.trainable_variables()
        accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs]
        self.zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]
        gvs = self.train_op.compute_gradients(loss, tvs)
        self.accum_ops = [accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(gvs)]
        self.actor_train_step = self.train_op.apply_gradients([(accum_vars[i], gv[1]) for i, gv in enumerate(gvs)])

    def _build_critic(self):
        self.R = tf.placeholder(tf.float32)

        cW1 = tf.Variable(tf.truncated_normal([self.Nfeature, self.Nhidden_unit], stddev=0.1 / np.sqrt(float(self.Nfeature))))
        cb1 = tf.Variable(tf.zeros([1, self.Nhidden_unit]))
        ch1 = tf.add(tf.matmul(self.input_docs, cW1), cb1)
        ch1 = tf.nn.relu(ch1)

        # Second layer -- linear classifier for action logits
        cW2 = tf.Variable(tf.truncated_normal([self.Nhidden_unit, 1], stddev=0.1 / np.sqrt(float(self.Nhidden_unit))))
        cb2 = tf.Variable(tf.zeros([1]))
        ch2 = tf.add(tf.matmul(ch1, cW2), cb2)
        ch2 = tf.transpose(ch2)
        self._estimate_advantage = tf.reduce_sum(ch2, axis=1)
        loss = tf.reduce_mean(tf.squared_difference(self._estimate_advantage, self.R))
        self.critic_train_step = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def update_actor_critic(self, query, ranklist, rewards, dataset):
        feature_matrix = dataset.get_all_features_by_query(query)
        ndoc = len(ranklist)
        lenghth = min(self.Lenepisode, ndoc)


        # print(self.sess.run([self.prob], feed_dict={self.input_docs: feature_matrix[ranklist[:10]]}))

        self.sess.run(self.zero_ops)

        for pos in range(lenghth):
            estimate_advantage = self.sess.run([self._estimate_advantage],
                                               feed_dict={self.input_docs: feature_matrix[ranklist[pos:]]})[0][0]
            advantage = rewards[pos] - estimate_advantage
            self.sess.run([self.accum_ops], feed_dict={self.input_docs: feature_matrix[ranklist[pos:]],
                                                       self.position: [0],
                                                       self.doc_length: len(ranklist[pos:]),
                                                       self.advantage: advantage})
        self.sess.run([self.actor_train_step])

        for pos in range(lenghth):
            self.sess.run([self.critic_train_step],
                          feed_dict={self.input_docs: feature_matrix[ranklist[pos:]],
                                     self.R: rewards[pos]})



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
            probabilities = self.softmax(scoretmp)

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
        return result
