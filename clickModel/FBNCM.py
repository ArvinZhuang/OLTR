import numpy as np
from pathlib import Path
# from clickModel.AbstractClickModel import AbstractClickModel
from clickModel.NCM import NCM
import bz2
import pickle
import tensorflow as tf
from keras.models import load_model, Model
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector, Concatenate

from keras.optimizers import Adadelta, Adam
from keras import backend as K
from utils import utility


class FBNCM(NCM):
    def __init__(self, n_a, q_dim, d_dim, dataset):
        super().__init__(n_a, q_dim, d_dim)
        self.dataset = dataset
        self.name = "FBNCM"

    def initial_representation(self, click_log):
        print("{} processing log.......".format(self.name))
        dataset_size = click_log.shape[0]

        for line in range(dataset_size):
            qid = click_log[line][0]
            docIds = click_log[line][1:11]
            # clicks = click_log[line][11:21]

            if qid not in self.query_rep.keys():
                feature_matrix = self.dataset.get_all_features_by_query(qid)
                self.query_rep[qid] = np.mean(feature_matrix, axis=0)
                self.doc_rep[qid] = {}

            for rank in range(10):
                docid = docIds[rank]
                if docid not in self.doc_rep[qid].keys():
                    self.doc_rep[qid][docid] = self.dataset.get_features_by_query_and_docid(qid, int(docid))

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
            docid = docids[rank]
            if docid not in self.doc_rep[qid].keys():
                D[0][rank] = self.dataset.get_features_by_query_and_docid(qid, int(docid))
                print("find unseen document: ", qid, docid)
            else:
                D[0][rank] = np.array(self.doc_rep[qid][docids[rank]])

        pred = self.inference_model.predict([x0, a0, c0, D, i0, q0])
        return np.array(pred)[:, 0, 0]


    def save_training_tfrecord(self, train_log, path, simulator):
        # train_log = train_log.reshape(-1, self._batch_size, 21)
        print("writing {} for {}.......".format(path, simulator))
        writer = tf.io.TFRecordWriter(path, options='GZIP')

        num_session = 0

        input = np.zeros((11, self.rep_dim))
        i0 = np.zeros(1)
        q0 = np.zeros(self.q_dim)

        for session in train_log:
            qid = session[0]
            docids = session[1:11]
            clicks = session[11:21]
            q_rep = self.query_rep[qid]

            t0 = np.append(q_rep, np.append(i0, np.zeros(self.d_dim)))
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
                if not utility.send_progress("@arvin {} generate {}".format(self.name, path), num_session, 400000,
                                             "num_session {}".format(num_session)):
                    print("internet disconnect")
        writer.close()

    def make_sequence_example(self, inputs, labels):
        """Returns a SequenceExample for the given inputs and labels.

        Args:
          inputs: A list of input vectors. Each input vector is a list of floats.
          labels: A list of ints.

        Returns:
          A tf.train.SequenceExample containing inputs and labels.
        """
        input_features = [
            tf.train.Feature(float_list=tf.train.FloatList(value=input_))
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
            "inputs": tf.io.FixedLenSequenceFeature([self.rep_dim], dtype=tf.float32),
            "labels": tf.io.FixedLenSequenceFeature([1], dtype=tf.int64)
        }
        # decode the TFRecord
        _, example = tf.io.parse_single_sequence_example(serialized=example, sequence_features=sequence_features)

        return example['inputs'], example['labels']