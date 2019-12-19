import numpy as np
from clickModel.CM import CM


class UBM(CM):
    def __init__(self, pc=None, eta=0.5, alpha=1, beta=1):
        self.name = 'UBM'
        self.alpha = alpha
        self.beta = beta
        self.pc = pc
        self.prr = np.power(np.divide(0.9, np.arange(1.0, 11.0)), eta)
        self.pr = np.arange(.55, 1.05, 0.05)[::-1]

        self.attract_parameters = {}
        self.rank_parameters = {}


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


    def _init_parameters(self, click_log):
        print("{} processing log.......".format(self.name))
        dataset_size = click_log.shape[0]
        for line in range(dataset_size):
            qid = click_log[line][0]
            docIds = click_log[line][1:11]

            if qid not in self.attract_parameters.keys():
                self.attract_parameters[qid] = {}

            doc_attract = self.attract_parameters[qid]
            for rank in len(docIds):
                docID = docIds[rank]
                if docID not in doc_attract.keys():
                    doc_attract[docID] = 0.2

        for rank in range(1, 11):
            self.rank_parameters[rank] = {}
            for i in range(1, rank + 1):
                self.rank_parameters[rank][i] = 0.5

