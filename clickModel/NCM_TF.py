from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
# from clickModel.AbstractClickModel import AbstractClickModel
from clickModel.CM import CM
import bz2
import pickle
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class NCM_TF(CM):

    def __init__(self, batch_size, lstm_num_hidden, num_of_dims, lstm_num_layers):
        super().__init__()
        self.name = 'NCM'
        self.query_rep = {}
        self.doc_rep = {}

        self._lstm_num_hidden = lstm_num_hidden
        self._lstm_num_layers = lstm_num_layers
        self._batch_size = batch_size
        self._representations_dims = num_of_dims

        # Initialization:
        self._inputs = tf.placeholder(tf.float32,
                                      shape=[self._batch_size, 11, self._representations_dims],
                                      name='inputs')
        self._targets = tf.placeholder(tf.float32,
                                       shape=[self._batch_size, 10],
                                       name='targets')

        self._targets_rshaped = tf.reshape(self._targets, [-1, 1])

        with tf.variable_scope('model'):
            self._logits_per_step = self._build_model()
            self._probabilities = self.probabilities()
            self._predictions = self.predictions()
            self._loss = self._compute_loss()


    def _build_model(self):

        with tf.variable_scope('states'):
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

            stacked_cells = tf.nn.rnn_cell.MultiRNNCell(
                [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(num_units=self._lstm_num_hidden, forget_bias=1.0),
                                    output_keep_prob=self.keep_prob)
                 for _ in range(self._lstm_num_layers)])

            self._state_placeholder = tf.placeholder(tf.float32,
                                                     [self._lstm_num_layers, 2, None, self._lstm_num_hidden])

            l = tf.unstack(self._state_placeholder, axis=1)

            self._rnn_tuple_state = tuple([tf.nn.rnn_cell.LSTMStateTuple(l[idx][0], l[idx][1])
                                           for idx in range(self._lstm_num_layers)])

            outputs, self._states = tf.nn.dynamic_rnn(cell=stacked_cells,
                                                      inputs=self._inputs,
                                                      initial_state=self._rnn_tuple_state,
                                                      dtype=tf.float32)

            # since for the first we only have s_0 and no click prediction
            outputs_ = outputs[:, 1:, :]

            outputs_rshaped = tf.reshape(tensor=outputs_, shape=[-1, self._lstm_num_hidden])

        with tf.variable_scope("predictions"):
            W_out = tf.get_variable("W_out",
                                    shape=[self._lstm_num_hidden, 1],
                                    initializer=tf.variance_scaling_initializer())

            b_out = tf.get_variable("b_out", shape=[1],
                                    initializer=tf.constant_initializer(0.0))

            predictions = tf.nn.bias_add(tf.matmul(outputs_rshaped, W_out), b_out)

        return predictions

    def _compute_loss(self):
        # Returns the log-likelihood

        with tf.name_scope('log_likelihood'):
            loss = tf.reduce_mean(tf.log(tf.multiply(self._probabilities, self._targets_rshaped)
                                         + tf.multiply((1 - self._targets_rshaped), (1 - self._probabilities))))

        return loss

    def probabilities(self):
        # Returns the normalized per-step probabilities
        probabilities = tf.nn.sigmoid(self._logits_per_step)
        return probabilities

    def predictions(self):
        # Returns the per-step predictions
        predictions = tf.round(self._probabilities)
        return predictions

    def train(self, X, Y):
        # train_log = train_log.reshape(-1, self._batch_size, 21)

        global_step = tf.Variable(0, trainable=False, dtype=tf.int32, name='global_step')
        starter_learning_rate = 0.01

        optimizer = tf.train.AdadeltaOptimizer(starter_learning_rate, epsilon=1e-06)

        # Compute the gradients for each variable
        grads_and_vars = optimizer.compute_gradients(-self._loss)

        # gradient clipping
        grads, variables = zip(*grads_and_vars)
        grads_clipped, _ = tf.clip_by_global_norm(grads, clip_norm=1)
        apply_gradients_op = optimizer.apply_gradients(zip(grads_clipped, variables), global_step=global_step)

        # start the Session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())


        # for batch_session in train_log:
        # inputs, targets = self._get_batch_train_sample(batch_session)

        # get the loss and the probabilities that the model outputs
        for iter in range(1000):
            _, loss_, pred, probs = self.sess.run([apply_gradients_op, self._loss, self._predictions, self._probabilities],
                                             feed_dict={self._inputs: X,
                                                        self._targets: Y,
                                                        self._state_placeholder: np.zeros((self._lstm_num_layers, 2,
                                                                                            self._batch_size,
                                                                                            self._lstm_num_hidden)),
                                                        self.keep_prob: 0.9})
            print(iter, loss_)

    def _get_batch_train_sample(self, batch_session):

        input = np.zeros((self._batch_size, 11, self._representations_dims))
        target = np.zeros((self._batch_size, 10))

        index = 0
        for session in batch_session:
            qid = session[0]
            docids = session[1:11]
            clicks = session[11:21]
            q_rep = self.query_rep[qid]

            t0 = np.append(q_rep, np.append(np.zeros(1), np.zeros(10240)))
            t1 = np.append(np.zeros(1024), np.append(np.zeros(1), self.doc_rep[qid][docids[0]]))
            input[index][0] = t0
            input[index][1] = t1

            for rank in range(1, 10):
                t = np.append(np.zeros(1024), np.append(clicks[rank-1],self.doc_rep[qid][docids[rank]]))
                input[index][rank] = t

            target[index] = clicks
            index += 1
        return input, target

    def _session_to_representations(self, session):
        qid = session[0]
        docids = session[1:11]
        clicks = session[11:21]
        q_rep = self.query_rep[qid]

        t0 = np.append(q_rep, np.append(np.zeros(1), np.zeros(10240)))
        t1 = np.append(np.zeros(1024), np.append(np.zeros(1), self.doc_rep[qid][docids[0]]))
        input[0] = t0
        input[1] = t1

        for rank in range(2, 11):
            t = np.append(np.zeros(1024), np.append(clicks[rank - 2], self.doc_rep[qid][docids[rank - 1]]))
            input[rank] = t

        target = clicks

    def save_training_set(self, train_log, path):
        # train_log = train_log.reshape(-1, self._batch_size, 21)
        train_size = train_log.shape[0]
        training_inputs = []
        traninig_labels = []

        input = np.zeros((11, self._representations_dims))

        i = 0
        for session in train_log:
            qid = session[0]
            docids = session[1:11]
            clicks = session[11:21]
            q_rep = self.query_rep[qid]

            t0 = np.append(q_rep, np.append(np.zeros(1), np.zeros(10240)))
            t1 = np.append(np.zeros(1024), np.append(np.zeros(1), self.doc_rep[qid][docids[0]]))
            input[0] = t0
            input[1] = t1

            for rank in range(2, 11):
                t = np.append(np.zeros(1024), np.append(clicks[rank - 2], self.doc_rep[qid][docids[rank-1]]))
                input[rank] = t

            target = clicks

            training_inputs.append(input)
            traninig_labels.append(target)
            i += 1
            print(i/train_size)



        with bz2.BZ2File(path+'Xv2.txt', 'w') as sfile:
            pickle.dump(np.array(training_inputs), sfile)
        with bz2.BZ2File(path+'Yv2.txt', 'w') as sfile:
            pickle.dump(np.array(traninig_labels), sfile)






    def initial_representation(self, click_log):
        print("{} processing log.......".format(self.name))
        dataset_size = click_log.shape[0]

        for line in range(dataset_size):
            qid = click_log[line][0]
            docIds = click_log[line][1:11]
            clicks = click_log[line][11:21]

            if qid not in self.query_rep.keys():
                self.query_rep[qid] = np.zeros(1024)
                self.doc_rep[qid] = {}
            clicks = clicks.astype(np.int)
            pattern_index = clicks.dot(1 << np.arange(clicks.shape[-1] - 1, -1, -1))
            self.query_rep[qid][pattern_index] += 1

            for rank in range(10):
                docid = docIds[rank]
                if docid not in self.doc_rep[qid].keys():
                    self.doc_rep[qid][docid] = np.zeros(1024*10)
                self.doc_rep[qid][docid][rank * 1024 + pattern_index] += 1

