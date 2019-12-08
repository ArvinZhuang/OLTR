import numpy as np
# from clickModel.AbstractClickModel import AbstractClickModel
from clickModel.CM import CM

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class LSTM(CM):
    def __init__(self, num_feature, num_hidden_units, train_set, learning_rate=0.001):
        self.name = "LSTM_models"
        self.lr = learning_rate
        self.num_feature = num_feature
        self.num_hidden_units = num_hidden_units
        self.train_set = train_set

        # Define weights
        self.weights = {
            # (10, num_hidden_units)
            'in': tf.Variable(tf.random_normal([self.num_feature, self.num_hidden_units])),
            # (num_hidden_units, 2)
            'out': tf.Variable(tf.random_normal([self.num_hidden_units, 2]))
        }
        self.biases = {
            # (num_hidden_units, )
            'in': tf.Variable(tf.constant(0.1, shape=[self.num_hidden_units, ])),
            # (2, )
            'out': tf.Variable(tf.constant(0.1, shape=[2, ]))
        }

        self.x = tf.placeholder(tf.float32, [None, 10, self.num_feature])
        self.y = tf.placeholder(tf.float32, [None, 10, 2])

        self.prediction, self.cell = self._RNN(self.x)
        self.output = self._predict(self.x, self.cell)

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

        init_state = cell.zero_state(180, dtype=tf.float32)

        outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)

        outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))

        results = tf.matmul(outputs, self.weights['out']) + self.biases['out']

        return results, cell

    def _predict(self, X, cell):
        X = tf.reshape(X, [-1, self.num_feature])

        X = tf.cast(X, tf.float32)

        X_in = tf.matmul(X, self.weights['in']) + self.biases['in']

        X_in = tf.reshape(X_in, [-1, 10, self.num_hidden_units])

        init_state = cell.zero_state(1, dtype=tf.float32)

        outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)

        outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))

        results = tf.matmul(outputs, self.weights['out']) + self.biases['out']

        return results

    def train(self, click_log):
        clicks = click_log[:, 11:]
        features = self._sessions_to_features(click_log)
        click_bitmap = self._clicks_to_bitmap(clicks)
        self.sess.run([self.train_op], feed_dict={
            self.x: features,
            self.y: click_bitmap,
        })

    def _clicks_to_bitmap(self, clicks):
        click_bitmap = np.zeros((1, 10, 2))
        clicks = clicks.astype(np.int)
        for i in range(clicks.shape[0]):
            sess_clicks = clicks[i].reshape(-1)
            click_label_flip = 1 - sess_clicks
            sess_clicks = np.vstack((sess_clicks, click_label_flip)).T
            click_bitmap = np.vstack((click_bitmap, np.array([sess_clicks])))
        return click_bitmap[1:]

    # def get_click_probs(self, session):
    #     logits = tf.nn.softmax(self.prediction)
    #     click_probs = self.sess.run(logits, feed_dict={self.x: session})
    #
    #     return click_probs.reshape(-1, 2)[:, 0]

    def get_click_probs(self, session):
        logits = tf.nn.softmax(self.output)
        click_probs = self.sess.run(logits, feed_dict={self.x: session})

        return click_probs.reshape(-1, 2)[:, 0]


    def get_MSE(self, test_click_log, dataset, simulator):
        print(self.name, "computing MSE")
        MSE = np.zeros(10)
        size = test_click_log.shape[0]
        for i in range(size):
            # print(i)
            session = test_click_log[i][:11]
            features = self._sessions_to_features(np.array([session]))
            click_probs = self.get_click_probs(features)
            real_click_probs = simulator.get_real_click_probs(session, dataset)
            MSE += np.square(click_probs - real_click_probs)

        return MSE/size

    def get_perplexity(self, test_click_log):
        perplexity = np.zeros(10)
        size = test_click_log.shape[0]
        for i in range(size):
            session = test_click_log[i][:11]
            click_label = test_click_log[i][11:]
            features = self._sessions_to_features(np.array[session])[0]
            click_probs = self.get_click_probs(features)
            for rank, click_prob in enumerate(click_probs):
                if click_label[rank] == '1':
                    p = click_prob
                else:
                    p = 1 - click_prob
                perplexity[rank] += np.log2(p)

        perplexity = [2 ** (-x / size) for x in perplexity]
        return perplexity

    def _sessions_to_features(self, sessions):
        qids = sessions[:, 0]
        features = np.zeros((1, 10, 700))
        for i in range(qids.shape[0]):
            docids = sessions[i][1:11].astype(np.int)
            features = np.vstack((features, np.array([self.train_set.get_all_features_by_query(qids[i])[docids]])))
        return features[1:]
