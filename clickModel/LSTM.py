import numpy as np
from clickModel.AbstractClickModel import AbstractClickModel
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class LSTM(AbstractClickModel):
    def __init__(self, num_feature, num_hidden_units, train_set, learning_rate=0.001):
        self.name = "LSTM"
        self.lr = learning_rate
        self.num_feature = num_feature
        self.num_hidden_units = num_hidden_units
        self.train_set = train_set

        # Define weights
        self.weights = {
            # (10, num_hidden_units)
            'in': tf.Variable(tf.random_normal([self.num_feature, self.num_hidden_units])),
            # (num_hidden_units, 1)
            'out': tf.Variable(tf.random_normal([self.num_hidden_units, 2]))
        }
        self.biases = {
            # (num_hidden_units, )
            'in': tf.Variable(tf.constant(0.1, shape=[self.num_hidden_units, ])),
            # (1, )
            'out': tf.Variable(tf.constant(0.1, shape=[2, ]))
        }

        self.x = tf.placeholder(tf.float32, [None, 10, self.num_feature])
        self.y = tf.placeholder(tf.float32, [None, 2])

        self.prediction = self._RNN(self.x)
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.prediction, labels=self.y))
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.cost)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _RNN(self, X):

        X = tf.reshape(X, [-1, self.num_feature])

        X = tf.cast(X, tf.float32)

        X_in = tf.matmul(X, self.weights['in']) + self.biases['in']

        X_in = tf.reshape(X_in, [-1, 10, self.num_hidden_units])

        cell = tf.nn.rnn_cell.BasicLSTMCell(self.num_hidden_units, forget_bias=1.0, state_is_tuple=True)
        init_state = cell.zero_state(1, dtype=tf.float32)

        outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)

        outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))

        results = tf.matmul(outputs, self.weights['out']) + self.biases['out']

        return results


    def train(self, click_log):

        for log in click_log:
            qid = log[0]
            docids = log[1:11]
            clicks = log[11:]
            features = []
            for docid in docids:
                features.append(self.dataset.get_features_by_query_and_docid(qid, docid))
            np.array([features])
            click_bitmap = self._clicks_to_bitmap(clicks)

            self.sess.run([self.train_op], feed_dict={
                self.x: features,
                self.y: click_bitmap,
            })

    def _clicks_to_bitmap(self, clicks):
        last_click = np.where(clicks)[0][-1]
        click_label = clicks[:last_click+1]
        click_label_flip = 1 - clicks[:last_click+1]
        click_bitmap = np.vstack((click_label, click_label_flip)).T
        return click_bitmap

    def get_click_probs(self, session):
        logits = tf.nn.softmax(self.prediction)
        click_probs = self.sess.run(logits, feed_dict={self.x: session})

        return click_probs.reshape(-1, 2)[:, 0]
