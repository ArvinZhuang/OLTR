import numpy as np
# from clickModel.AbstractClickModel import AbstractClickModel
from clickModel.CM import CM

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam


# from tensorflow.keras.metrics import categorical_accuracy


class LSTMv2(CM):
    def __init__(self, num_feature,
                 num_hidden_units,
                 dataset,
                 learning_rate=0.001,
                 batch_size=32,
                 epoch=20,
                 steps_per_epoch=None,
                 model=None,
                 ):
        self.name = "LSTM_models"
        self.lr = learning_rate
        self.num_feature = num_feature
        self.num_hidden_units = num_hidden_units
        self.dataset = dataset
        self.batch_size = batch_size
        self.epoch = epoch
        self.steps_per_epoch = steps_per_epoch

        if model is not None:
            self.model = model
            print(self.model.summary())
        else:
            self.model = tf.keras.Sequential()
            # input shape (None, 10, 700), output shape
            self.model.add(layers.Bidirectional(layers.LSTM(num_hidden_units, return_sequences=True),
                                                input_shape=(10, num_feature)))

            # forward_layer = layers.LSTM(num_hidden_units, return_sequences=True)
            # backward_layer = layers.LSTM(num_hidden_units, activation='relu', return_sequences=True,
            #                      go_backwards=True)
            #
            # self.model.add(layers.Bidirectional(forward_layer, backward_layer=backward_layer,
            #                         input_shape=(10, num_feature)))

            # output shape (None, 10, 1)
            # self.model.add(layers.Dense(32, activation='relu'))
            self.model.add(layers.Dense(1, activation='sigmoid'))
            print(self.model.summary())

            adam = Adam(learning_rate)

            self.model.compile(optimizer=adam,
                               loss='binary_crossentropy',
                               metrics=["accuracy"])

    def train(self, click_log):
        # print("training set shape:", tf.data.experimental.cardinality(click_log))
        click_log = click_log.map(self._read_tfrecord)
        click_log = click_log.repeat(1)
        click_log = click_log.shuffle(2048)
        click_log = click_log.batch(self.batch_size, drop_remainder=True)

        # test_log = test_log.map(self._read_tfrecord)
        # test_log = test_log.batch(self.batch_size, drop_remainder=False)
        self.model.fit(click_log, epochs=self.epoch, steps_per_epoch=self.steps_per_epoch)

        # clicks = click_log[:, 11:]
        # for i in range(click_log.shape[0]):
        #     features = self._sessions_to_features(np.array([click_log[i]]))
        #     click_bitmap = self._clicks_to_bitmap(np.array([clicks[i]]))
        #
        #     self.model.fit(features, click_bitmap,verbose=0)
        #                    # validation_data=(x_test, y_test),
        #                    # batch_size=1,
        #                    # epochs=1,
        #                    # validation_split=0.1)

        # clicks = click_log[:, 11:]
        #
        # features = self._sessions_to_features(click_log)
        # click_bitmap = self._clicks_to_bitmap(clicks)
        #
        # #
        # self.model.fit(features, click_bitmap, verbose=0,
        #         batch_size=180,)
        #         # epochs=1,)

    def _clicks_to_bitmap(self, clicks):
        click_bitmap = np.zeros((1, 10, 1))
        clicks = clicks.astype(np.int)
        for i in range(clicks.shape[0]):
            sess_clicks = clicks[i].reshape(-1, 1)
            click_bitmap = np.vstack((click_bitmap, np.array([sess_clicks])))
        return click_bitmap[1:]

    def get_click_probs(self, session):

        predicts = self.model.predict(session)

        return predicts[0].reshape(-1)

    def get_MSE(self, test_click_log, dataset, simulator):
        print(self.name, "computing MSE")
        MSE = np.zeros(10)
        size = test_click_log.shape[0]
        for i in range(size):
            # print(i)
            if i % 1000 == 0:
                print("\r", end='')
                print(str(i / size) + " complete!", end="", flush=True)
            session = test_click_log[i]
            features = self._sessions_to_features(np.array([session]))
            click_probs = self.get_click_probs(features)
            real_click_probs = simulator.get_real_click_probs(session, dataset)
            # print("predicts: ", click_probs)
            # print("real: ", real_click_probs)
            # print()
            MSE += np.square(click_probs - real_click_probs)

        return MSE / size

    def get_perplexity(self, test_click_log):
        print(self.name, "computing perplexity")
        perplexity = np.zeros(10)
        size = test_click_log.shape[0]
        for i in range(size):
            if i % 1000 == 0:
                print("\r", end='')
                print(str(i / size) + " complete!", end="", flush=True)
            session = test_click_log[i][:11]
            click_label = test_click_log[i][11:]
            features = self._sessions_to_features(np.array([session]))
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
            features = np.vstack((features, np.array([self.dataset.get_all_features_by_query(qids[i])[docids]])))
        return features[1:]

    def _read_tfrecord(self, example):
        sequence_features = {
            "inputs": tf.io.FixedLenSequenceFeature([700], dtype=tf.float32),
            "labels": tf.io.FixedLenSequenceFeature([1], dtype=tf.int64)
        }
        # decode the TFRecord
        _, example = tf.io.parse_single_sequence_example(serialized=example, sequence_features=sequence_features)

        return example['inputs'], example['labels']
