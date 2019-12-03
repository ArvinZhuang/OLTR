import numpy as np
# from clickModel.AbstractClickModel import AbstractClickModel
from clickModel.CM import CM

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_accuracy


class LSTMv2(CM):
    def __init__(self, num_feature, num_hidden_units, train_set, learning_rate=0.001):
        self.name = "LSTM"
        self.lr = learning_rate
        self.num_feature = num_feature
        self.num_hidden_units = num_hidden_units
        self.train_set = train_set

        self.model = tf.keras.Sequential()
        # input shape (None, 10, 700), output shape
        self.model.add(layers.LSTM(num_hidden_units, input_shape=(10, num_feature),
                                   return_sequences=True,
                                   ))
        # output shape (None, 10, 1)
        self.model.add(layers.Dense(1, activation='sigmoid'))
        print(self.model.summary())

        adam = Adam(learning_rate)

        self.model.compile(optimizer=adam,
                           loss='binary_crossentropy',
                           metrics=[categorical_accuracy])


    def train(self, click_log, trainset, epoch):

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
        self.model.fit(trainset, epochs=epoch)


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
            print(str(i / size) + " complete!", end='\r')
            session = test_click_log[i][:11]
            features = self._sessions_to_features(np.array([session]))
            click_probs = self.get_click_probs(features)
            real_click_probs = simulator.get_real_click_probs(session, dataset)
            # print("predicts: ", click_probs)
            # print("real: ", real_click_probs)
            # print()
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
