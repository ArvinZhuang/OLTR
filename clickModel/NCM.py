import numpy as np
# from clickModel.AbstractClickModel import AbstractClickModel
from clickModel.CM import CM

import tensorflow as tf

class NCM(CM):
    def __init__(self):
        self.name = 'NCM'
        self.query_rep = {}
        self.doc_rep = {}


    def initial_representation(self, click_log):
        print("{} processing log.......".format(self.name))
        dataset_size = click_log.shape[0]

        for line in range(dataset_size):
            qid = click_log[line][0]
            docIds = click_log[line][1:11]
            clicks = click_log[line][11:]

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

