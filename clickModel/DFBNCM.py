import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";

# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "2";
import numpy as np
from pathlib import Path
# from clickModel.AbstractClickModel import AbstractClickModel
from clickModel.FBNCM import FBNCM
import bz2
import pickle
import tensorflow as tf
from keras.models import load_model, Model
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector, Concatenate

from keras.optimizers import Adadelta, Adam
from keras import backend as K
from utils import utility


class DFBNCM(FBNCM):
    def __init__(self, n_a, dq_dim, dd_dim, fq_dim, fd_dim, dataset, model=None):
        super().__init__(n_a, dq_dim+fq_dim, dd_dim+fd_dim, dataset, model)
        self.name = "DFBNCM"
        self.dq_dim = dq_dim
        self.dd_dim = dd_dim
        self.fq_dim = fq_dim
        self.fd_dim = fd_dim


    def initial_representation(self, click_log):
        print("{} processing log.......".format(self.name))
        dataset_size = click_log.shape[0]

        for line in range(dataset_size):
            qid = click_log[line][0]
            docIds = click_log[line][1:11]
            clicks = click_log[line][11:21]

            clicks = clicks.astype(np.int)
            pattern_index = clicks.dot(1 << np.arange(clicks.shape[-1] - 1, -1, -1))  # binary to decimal

            if qid not in self.query_rep.keys():
                feature_matrix = self.dataset.get_all_features_by_query(qid)
                feature_rep = np.mean(feature_matrix, axis=0)
                distribution_rep = np.zeros(self.dq_dim)

                distribution_rep[pattern_index] += 1

                self.query_rep[qid] = np.append(feature_rep, distribution_rep)
                self.doc_rep[qid] = {}
            else:
                self.query_rep[qid][self.fq_dim+pattern_index] += 1

            for rank in range(10):
                docid = docIds[rank]
                if docid not in self.doc_rep[qid].keys():
                    feature_rep = self.dataset.get_features_by_query_and_docid(qid, int(docid))
                    distribution_rep = np.zeros(self.dd_dim)
                    distribution_rep[rank * self.dq_dim + pattern_index] += 1
                    self.doc_rep[qid][docid] = np.append(feature_rep, distribution_rep)
                else:
                    self.doc_rep[qid][docid][self.fd_dim + rank * self.dq_dim + pattern_index] += 1

    def get_click_probs(self, session):
        qid = session[0]
        docids = session[1:11]
        clicks = session[11:21]
        if qid not in self.query_rep.keys():
            feature_matrix = self.dataset.get_all_features_by_query(qid)
            feature_rep = np.mean(feature_matrix, axis=0)
            distribution_rep = np.zeros(self.dq_dim)

            self.query_rep[qid] = np.append(feature_rep, distribution_rep)
            self.doc_rep[qid] = {}
        else:
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
                feature_rep = self.dataset.get_features_by_query_and_docid(qid, int(docid))
                distribution_rep = np.zeros(self.dd_dim, dtype=int)
                D[0][rank] = np.append(feature_rep, distribution_rep)
            else:
                D[0][rank] = np.array(self.doc_rep[qid][docids[rank]])

        pred = self.inference_model.predict([x0, a0, c0, D, i0, q0])
        return np.array(pred)[:, 0, 0]
