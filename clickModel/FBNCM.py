import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
#
# # The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "2";
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
    def __init__(self, n_a, q_dim, d_dim, dataset, model=None):
        super().__init__(n_a, q_dim, d_dim, model)
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
        if qid not in self.query_rep.keys():
            feature_matrix = self.dataset.get_all_features_by_query(qid)
            self.query_rep[qid] = np.mean(feature_matrix, axis=0)
            self.doc_rep[qid] = {}
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
            else:
                D[0][rank] = np.array(self.doc_rep[qid][docids[rank]])
        pred = self.inference_model.predict([x0, a0, c0, D, i0, q0])

        return np.array(pred)[:, 0, 0]