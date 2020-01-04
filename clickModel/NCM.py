import numpy as np
from pathlib import Path
# from clickModel.AbstractClickModel import AbstractClickModel
from clickModel.CM import CM
import bz2
import pickle
import tensorflow as tf
from keras.models import load_model, Model
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector, Concatenate

from keras.optimizers import Adadelta, Adam
from keras import backend as K
from utils import utility




class NCM(CM):
    def __init__(self, n_a, q_dim, d_dim):
        super().__init__()
        self.name = 'NCM'
        self.n_a = n_a
        self.q_dim = q_dim
        self.d_dim = d_dim
        self.rep_dim = q_dim + 1 + d_dim
        self.reshapor = Reshape((1, self.rep_dim))  # Used in Step 2.B of djmodel(), below
        self.LSTM_cell = LSTM(n_a, return_state=True)  # Used in Step 2.C
        self.densor = Dense(1, activation='sigmoid')
        self.model = self._build_model()
        opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)
        self.model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        # print(self.model.summary())

        self.query_rep = {}
        self.doc_rep = {}


    def _build_model(self):
        # Define the input layer and specify the shape
        X = Input(shape=(11, self.rep_dim))
        a0 = Input(shape=(self.n_a,), name='a0')
        c0 = Input(shape=(self.n_a,), name='c0')
        a = a0
        c = c0
        outputs = []

        for t in range(11):
            x = Lambda(lambda X: X[:, t, :])(X)
            x = self.reshapor(x)
            a, _, c = self.LSTM_cell(inputs=x, initial_state=[a, c])
            if t >= 1:
                out = self.densor(a)
                outputs.append(out)
        model = Model(inputs=[X, a0, c0], outputs=outputs)
        self.test_model = Model(inputs=[X, a0, c0], outputs=outputs)
        return model


    def _build_inference_model(self):
        x0 = Input(shape=(1, self.rep_dim))

        # Define s0, initial hidden state for the decoder LSTM
        a0 = Input(shape=(self.n_a,), name='a0')
        c0 = Input(shape=(self.n_a,), name='c0')
        i0 = Input(shape=(1,), name='i0')
        q0 = Input(shape=(self.q_dim,), name='q0')
        D = Input(shape=(10, self.d_dim), name='D')
        a = a0
        c = c0
        x = x0


        ### START CODE HERE ###
        # Step 1: Create an empty list of "outputs" to later store your predicted values (≈1 line)
        outputs = []
        # Step 2: Loop over Ty and generate a value at every time step
        for t in range(11):
            # Step 2.A: Perform one step of LSTM_cell (≈1 line)
            a, _, c = self.LSTM_cell(inputs=x, initial_state=[a, c])

            # Step 2.B: Apply Dense layer to the hidden state output of the LSTM_cell (≈1 line)
            if t < 10:
                x = Lambda(lambda D: D[:, t, :])(D)
            if t > 0:
                out = self.densor(a)

                outputs.append(out)
                if t < 10:
                    x = Lambda(self._concatebate)([q0, out, x])
            else:
                x = Lambda(self._concatebate)([q0, i0, x])

        # Step 3: Create model instance with the correct "inputs" and "outputs" (≈1 line)
        inference_model = Model(inputs=[x0, a0, c0, D, i0, q0], outputs=outputs)
        return inference_model

    def _concatebate(self, rep):
        q = rep[0]
        out = rep[1]
        # print("1", out)
        # out = K.round(out)
        # print("2", out)
        x = rep[2]
        x = K.concatenate((q, out, x))
        x = RepeatVector(1)(x)
        return x


    def train_with_numpy(self, X, Y):
        a0 = np.zeros((X.shape[0], self.n_a))
        c0 = np.zeros((X.shape[0], self.n_a))
        self.model.fit([X, a0, c0], list(Y), batch_size=30, epochs=300)
        self.inference_model = self._build_inference_model()

    def train_tfrecord(self, path, batch_size=32, epoch=5, steps_per_epoch=1):
        print("start")

        tfrecord = tf.data.TFRecordDataset(path)
        tfrecord = tfrecord.map(self._read_tfrecord)
        tfrecord = tfrecord.repeat(epoch)
        # tfrecord = tfrecord.shuffle(batch_size*10)
        tfrecord = tfrecord.batch(batch_size, drop_remainder=False)

        a0 = np.zeros((batch_size, self.n_a))
        c0 = np.zeros((batch_size, self.n_a))
        i = 0
        for batch in tfrecord:
            i += 1

            X, Y = batch
            Y = tf.reshape(tf.transpose(Y), [10, -1, 1])

            Y = tf.reshape(Y, (10, -1, 1))
            loss = self.model.fit([X, a0, c0], list(Y), steps_per_epoch=steps_per_epoch, verbose=0)

            trained = i * batch_size
            if trained % 6400 == 0:
                print("finished:", trained/(400000 * epoch), "loss:", loss.history["loss"][0])
                if not utility.send_progress("@arvin training {} model, file: {}".format(self.name, path),
                                             trained,
                                             400000 * epoch,
                                             "loss: " + str(loss.history["loss"][0])):
                    print("internet disconnect")

        self.inference_model = self._build_inference_model()


    def get_click_probs(self, session):
        qid = session[0]
        docids = session[1:11]
        clicks = session[11:21]
        q_rep = self.query_rep[qid]
        x0 = np.append(q_rep, np.append(np.zeros(1), np.zeros(self.d_dim)))
        x0 = x0.reshape((1, 1, -1))  # shape (1, 1, 11265)
        a0 = np.zeros((1, self.n_a))  # shape (1, 64)
        c0 = np.zeros((1, self.n_a))  # shape (1, 64)
        i0 = np.zeros((1, 1))  # shape (1, 1)
        q0 = np.zeros((1, self.q_dim))  # shape (1, 1024)

        D = np.zeros((1, 10, self.d_dim))  # shape (1, 1, 10240)
        for rank in range(10):
            D[0][rank] = np.array(self.doc_rep[qid][docids[rank]])

        pred = self.inference_model.predict([x0, a0, c0, D, i0, q0])
        return np.array(pred)[:, 0, 0]

    def initial_representation(self, click_log):
        print("{} processing log.......".format(self.name))
        dataset_size = click_log.shape[0]

        for line in range(dataset_size):
            qid = click_log[line][0]
            docIds = click_log[line][1:11]
            clicks = click_log[line][11:21]

            if qid not in self.query_rep.keys():
                self.query_rep[qid] = np.zeros(self.q_dim, dtype=int)
                self.doc_rep[qid] = {}
            clicks = clicks.astype(np.int)
            pattern_index = clicks.dot(1 << np.arange(clicks.shape[-1] - 1, -1, -1))  # binary to decimal
            self.query_rep[qid][pattern_index] += 1

            for rank in range(10):
                docid = docIds[rank]
                if docid not in self.doc_rep[qid].keys():
                    self.doc_rep[qid][docid] = np.zeros(self.d_dim, dtype=int)
                self.doc_rep[qid][docid][rank * self.q_dim + pattern_index] += 1

    def save_training_tfrecord(self, train_log, path, simulator):
        # train_log = train_log.reshape(-1, self._batch_size, 21)
        print("writing tfrecord file for {}.......".format(simulator))
        writer = tf.io.TFRecordWriter(path)

        num_session = 0

        input = np.zeros((11, self.rep_dim), dtype=int)
        i0 = np.zeros(1, dtype=int)
        q0 = np.zeros(self.q_dim, dtype=int)

        for session in train_log:
            qid = session[0]
            docids = session[1:11]
            clicks = session[11:21]
            q_rep = self.query_rep[qid]

            t0 = np.append(q_rep, np.append(i0, np.zeros(self.d_dim, dtype=int)))
            t1 = np.append(q0, np.append(i0, self.doc_rep[qid][docids[0]]))

            input[0] = t0
            input[1] = t1

            for rank in range(2, 11):
                t = np.append(q0, np.append(clicks[rank - 2], self.doc_rep[qid][docids[rank-1]]))
                input[rank] = t

            output = np.array(clicks).reshape((-1, 1))

            example = self.make_sequence_example(input, output)
            serialized = example.SerializeToString()
            writer.write(serialized)
            num_session += 1
            if num_session % 1000 == 0:
                if not utility.send_progress("@arvin {} generate {} model .tfrecord file".format(self.name, simulator), num_session, 400000,
                                             "train_set1_NCM"):
                    print("internet disconnect")
        writer.close()

    def save_training_numpy(self, train_log, path, simulator):
        # train_log = train_log.reshape(-1, self._batch_size, 21)
        print("writing numpy file.......")

        num_session = 0
        input = np.zeros((train_log.shape[0], 11, self.rep_dim))
        i0 = np.zeros(1)
        q0 = np.zeros(self.q_dim)
        d0 = np.zeros(self.d_dim)
        label = np.zeros((10, len(train_log), 1))

        for session in train_log:
            qid = session[0]
            docids = session[1:11]
            clicks = session[11:21]
            q_rep = self.query_rep[qid]

            t0 = np.append(q_rep, np.append(i0, d0))
            t1 = np.append(q0, np.append(i0, self.doc_rep[qid][docids[0]]))

            input[num_session][0] = t0
            input[num_session][1] = t1

            for rank in range(2, 11):
                t = np.append(q0, np.append(clicks[rank - 2], self.doc_rep[qid][docids[rank-1]]))
                input[num_session][rank] = t

            label[:, num_session, :] = clicks.reshape(10, 1)

            num_session += 1

            if num_session % 1000 == 0:
                print("\r", end='')
                print("num_of_writen:", num_session / 400000, end="", flush=True)
                if not utility.send_progress("@arvin generate {} model numpy file".format(simulator), num_session, 400000,
                                             "train_set1_NCM"):
                    print("internet disconnect")

        np.savez(path, input=input, label=label)
        # with bz2.BZ2File(path+"input.txt", 'w') as fp:
        #     pickle.dump(input, fp)
        # with bz2.BZ2File(path+"label.txt", 'w') as fp:
        #     pickle.dump(label, fp)


    def make_sequence_example(self, inputs, labels):
        """Returns a SequenceExample for the given inputs and labels.

        Args:
          inputs: A list of input vectors. Each input vector is a list of floats.
          labels: A list of ints.

        Returns:
          A tf.train.SequenceExample containing inputs and labels.
        """
        input_features = [
            tf.train.Feature(int64_list=tf.train.Int64List(value=input_))
            for input_ in inputs]
        label_features = [
            tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
            for label in labels]
        feature_list = {
            'inputs': tf.train.FeatureList(feature=input_features),
            'labels': tf.train.FeatureList(feature=label_features)
        }
        feature_lists = tf.train.FeatureLists(feature_list=feature_list)
        return tf.train.SequenceExample(feature_lists=feature_lists)

    def _read_tfrecord(self, example):
        sequence_features = {
            "inputs": tf.io.FixedLenSequenceFeature([self.rep_dim], dtype=tf.int64),
            "labels": tf.io.FixedLenSequenceFeature([1], dtype=tf.int64)
            # "init_state": tf.io.FixedLenSequenceFeature([], dtype=tf.int64),
            # "init_cell": tf.io.FixedLenSequenceFeature([], dtype=tf.int64)
        }
        # decode the TFRecord
        _, example = tf.io.parse_single_sequence_example(serialized=example, sequence_features=sequence_features)

        return example['inputs'], example['labels']