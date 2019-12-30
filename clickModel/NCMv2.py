import numpy as np
# from clickModel.AbstractClickModel import AbstractClickModel
from clickModel.CM import CM

import tensorflow as tf
from keras.models import load_model, Model
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector, Concatenate
from keras.initializers import glorot_uniform
from keras.utils import to_categorical
from keras.optimizers import Adadelta, Adam
from keras import backend as K




class NCMv2(CM):
    def __init__(self, n_a, rep_dim):
        self.name = 'NCMv2'
        super().__init__()
        self.n_a = n_a
        self.rep_dim = rep_dim
        self.reshapor = Reshape((1, self.rep_dim))  # Used in Step 2.B of djmodel(), below
        self.LSTM_cell = LSTM(n_a, return_state=True)  # Used in Step 2.C
        self.densor = Dense(1, activation='sigmoid')
        self.model = self._build_model()
        # print(self.model.summary())

        self.query_rep = {}
        self.doc_rep = {}


    def _build_model(self):
        # Define the input layer and specify the shape
        X = Input(shape=(11, self.rep_dim))

        # Define the initial hidden state a0 and initial cell state c0
        # using `Input`
        a0 = Input(shape=(self.n_a,), name='a0')
        c0 = Input(shape=(self.n_a,), name='c0')
        a = a0
        c = c0

        ### START CODE HERE ###
        # Step 1: Create empty list to append the outputs while you iterate (≈1 line)
        outputs = []

        # Step 2: Loop
        for t in range(11):
            # Step 2.A: select the "t"th time step vector from X.
            x = Lambda(lambda X: X[:, t, :])(X)
            # Step 2.B: Use reshapor to reshape x to be (1, n_values) (≈1 line)
            x = self.reshapor(x)
            # Step 2.C: Perform one step of the LSTM_cell
            a, _, c = self.LSTM_cell(inputs=x, initial_state=[a, c])
            if t >= 1:
                # Step 2.D: Apply densor to the hidden state output of LSTM_Cell
                out = self.densor(a)
                # Step 2.E: add the output to "outputs"
                outputs.append(out)

        # Step 3: Create model instance
        model = Model(inputs=[X, a0, c0], outputs=outputs)
        return model

    def _build_inference_model(self):
        x0 = Input(shape=(1, self.rep_dim))

        # Define s0, initial hidden state for the decoder LSTM
        a0 = Input(shape=(self.n_a,), name='a0')
        c0 = Input(shape=(self.n_a,), name='c0')
        i0 = Input(shape=(1,), name='i0')
        q0 = Input(shape=(1024,), name='q0')
        D = Input(shape=(10, 10240), name='D')
        a = a0
        c = c0
        x = x0


        ### START CODE HERE ###
        # Step 1: Create an empty list of "outputs" to later store your predicted values (≈1 line)
        outputs = []
        # Step 2: Loop over Ty and generate a value at every time step
        for t in range(11):
            # Step 2.A: Perform one step of LSTM_cell (≈1 line)
            a, _, c = self.LSTM_cell(x, initial_state=[a, c])

            # Step 2.B: Apply Dense layer to the hidden state output of the LSTM_cell (≈1 line)
            if t < 10:
                x = Lambda(lambda D: D[:, t, :])(D)
            if t > 0:
                out = self.densor(a)

                # Step 2.C: Append the prediction "out" to "outputs". out.shape = (None, 78) (≈1 line)
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


    def train(self, X, Y):
        Y = Y.T.reshape((10, -1, 1))
        # opt = Adadelta(clipnorm=1.)
        opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)
        self.model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        a0 = np.zeros((X.shape[0], self.n_a))
        c0 = np.zeros((X.shape[0], self.n_a))
        self.model.fit([X, a0, c0], list(Y), batch_size=774, epochs=300)
        self.inference_model = self._build_inference_model()



    def predict(self, session):
        print(session)
        qid = session[0]
        docids = session[1:11]
        clicks = session[11:21]
        q_rep = self.query_rep[qid]
        x0 = np.append(q_rep, np.append(np.zeros(1), np.zeros(10240)))
        x0 = x0.reshape((1, 1, -1))  # shape (1, 1, 11265)
        a0 = np.zeros((1, self.n_a))  # shape (1, 64)
        c0 = np.zeros((1, self.n_a))  # shape (1, 64)
        i0 = np.zeros((1, 1))  # shape (1, 1)
        q0 = np.zeros((1, 1024))  # shape (1, 1024)

        D = np.zeros((1, 10, 10240))  # shape (1, 1, 10240)
        for rank in range(10):
            D[0][rank] = np.array(self.doc_rep[qid][docids[rank]])

        pred = self.inference_model.predict([x0, a0, c0, D, i0, q0])
        print(pred)
        print(len(pred))




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

    def save_training_set(self, train_log, path):
        # train_log = train_log.reshape(-1, self._batch_size, 21)
        train_size = train_log.shape[0]
        writer = tf.io.TFRecordWriter("test.tfrecord")

        i = 0
        for session in train_log:
            qid = session[0]
            docids = session[1:11]
            clicks = session[11:21]
            q_rep = self.query_rep[qid]

            t0 = np.append(q_rep, np.append(np.zeros(1), np.zeros(10240)))
            t1 = np.append(np.zeros(1024), np.append(np.zeros(1), self.doc_rep[qid][docids[0]]))

            input = np.zeros((11, self.rep_dim))
            input[0] = t0
            input[1] = t1

            for rank in range(2, 11):
                t = np.append(np.zeros(1024), np.append(clicks[rank - 2], self.doc_rep[qid][docids[rank-1]]))
                input[rank] = t

            target = np.array(clicks).T.reshape((10, -1, 1))
            print(input.shape)

            example = self._make_sequence_example(input.tolist(), target.tolist())
            writer.write(example.SerializeToString())
            i += 1
            print(i/train_size)

    def _make_sequence_example(self, input, label):
        """Returns a SequenceExample for the given inputs and labels.

        Args:
          inputs: A list of input vectors. Each input vector is a list of floats.
          labels: A list of ints.

        Returns:
          A tf.train.SequenceExample containing inputs and labels.
        """
        print(len(input[0]))
        input_feature = tf.train.Feature(float_list=tf.train.FloatList(value=[input]))

        label_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))

        feature = {
            'input': input_feature,
            'label': label_feature
        }

        return tf.train.Example(features=tf.train.Features(feature=feature))

