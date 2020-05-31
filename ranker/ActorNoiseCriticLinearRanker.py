from ranker.ActorCriticLinearRanker import ActorCriticLinearRanker
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


class ActorNoiseCriticLinearRanker(ActorCriticLinearRanker):
    def __init__(self,
                 Nfeature,
                 Learningrate,
                 Nhidden_unit=256,
                 Lenepisode=10,
                 memory_size=100,
                 batch_size=1):
        super().__init__(Nfeature, Learningrate, Nhidden_unit, Lenepisode, memory_size, batch_size)

    def _build_critic(self):
        self.c = tf.placeholder(tf.float32)

        cW1 = tf.Variable(
            tf.truncated_normal([self.Nfeature, self.Nhidden_unit], stddev=0.1 / np.sqrt(float(self.Nfeature))))
        cb1 = tf.Variable(tf.zeros([1, self.Nhidden_unit]))
        ch1 = tf.add(tf.matmul(self.input_docs, cW1), cb1)
        ch1 = tf.nn.relu(ch1)

        # Second layer -- linear classifier for action logits
        cW2 = tf.Variable(tf.truncated_normal([self.Nhidden_unit, 1], stddev=0.1 / np.sqrt(float(self.Nhidden_unit))))
        cb2 = tf.Variable(tf.zeros([1]))
        ch2 = tf.add(tf.matmul(ch1, cW2), cb2)
        ch2 = tf.transpose(ch2)
        logits = tf.reduce_sum(ch2, axis=1)

        self._click_prob = tf.sigmoid(logits)

        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.c, logits=logits)

        self.critic_train_step = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def update_actor_critic(self, query, ranklist, rewards, dataset):
        feature_matrix = dataset.get_all_features_by_query(query)
        ndoc = len(ranklist)
        lenghth = min(self.Lenepisode, ndoc)

        # print(self.sess.run([self.prob], feed_dict={self.input_docs: feature_matrix[ranklist[:10]]}))

        self.sess.run(self.zero_ops)

        for pos in range(lenghth):
            click_prob = self.sess.run([self._click_prob],
                                       feed_dict={self.input_docs: feature_matrix[ranklist[pos:]]})[0][0]
            if rewards[pos] > 0:
                advantage = rewards[pos] * click_prob
            else:
                advantage = rewards[pos] * (1 - click_prob)
            self.sess.run([self.accum_ops], feed_dict={self.input_docs: feature_matrix[ranklist[pos:]],
                                                       self.position: [0],
                                                       self.doc_length: len(ranklist[pos:]),
                                                       self.advantage: advantage})
        self.sess.run([self.actor_train_step])

        for pos in range(lenghth):
            if rewards[pos] > 0:
                click_label = 1
            else:
                click_label = 0
            self.sess.run([self.critic_train_step],
                          feed_dict={self.input_docs: feature_matrix[ranklist[pos:]],
                                     self.c: click_label})
