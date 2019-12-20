import numpy as np
import copy
from clickModel.CM import CM


class UBM(CM):
    def __init__(self, pc=None, eta=0.5, iter=50):
        self.name = 'UBM'
        self.iter = iter
        self.pc = pc
        self.prr = np.power(np.divide(0.9, np.arange(1.0, 11.0)), eta)
        self.pr = np.arange(.55, 1.05, 0.05)[::-1]

        self.attr_parameters = {}
        self.exam_parameters = {}
        self.query_stat = {}


    def simulate(self, query, result_list, dataset):
        clicked_doc = []
        click_label = np.zeros(len(result_list))
        satisfied = True
        last_click = 0
        for i in range(0, len(result_list)):
            click_prob = np.random.rand()
            exam_prob = np.random.rand()
            docid = result_list[i]

            relevance = dataset.get_relevance_label_by_query_and_docid(query, docid)

            if exam_prob <= self.pr[i] * self.prr[last_click]:
                if click_prob <= self.pc[relevance]:
                    click_label[i] = 1
                    clicked_doc.append(result_list[i])
                    last_click = 0
                    continue
            last_click += 1
        return clicked_doc, click_label, satisfied

    def train(self, click_log):
        self._init_parameters(click_log)

        print("{} training.......".format(self.name))
        for i in self.iter:
            new_attr_params = copy.deepcopy(self.attr_parameters)

            for qid in self.attr_parameters.keys():
                for docid in self.attr_parameters[qid].keys():
                    numerator = 0
                    denominator = 0
                    attr = self.attr_parameters[qid][docid]
                    for rank, click, last_click in self.query_stat[qid][docid]:
                        if click == 1:
                            numerator += 1
                        else:
                            exam = self.exam_parameters[rank][last_click]
                            numerator += ((1 - exam) * attr) / (1 - exam * attr)

                        denominator += 1

                    new_attr_params[qid][docid] = numerator / denominator





    def _init_parameters(self, click_log):
        print("{} processing log.......".format(self.name))
        dataset_size = click_log.shape[0]
        for line in range(dataset_size):
            qid = click_log[line][0]
            docIds = click_log[line][1:11]
            clicks = click_log[line][11:]
            last_click = 0
            if qid not in self.attr_parameters.keys():
                self.attr_parameters[qid] = {}
                self.query_stat[qid] = {}

            doc_attract = self.attr_parameters[qid]
            doc_stat = self.query_stat[qid]

            for rank in range(len(docIds)):
                if clicks[rank] == 1:
                    last_click = rank + 1

                docID = docIds[rank]
                if docID not in doc_attract.keys():
                    doc_attract[docID] = 0.2
                    doc_stat[docID] = []

                doc_stat[docID].append((rank+1, clicks[rank], rank+1 - last_click))

        for rank in range(1, 11):
            self.exam_parameters[rank] = {}
            for i in range(1, rank + 1):
                self.exam_parameters[rank][i] = 0.5

