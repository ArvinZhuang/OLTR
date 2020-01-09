import numpy as np
from clickModel.CM import CM


class DCTR(CM):
    def __init__(self, pc=None, alpha=1, beta=1):
        self.name = 'DCTR'
        self.parameter_dict = {}
        self.stat_dict = {}
        self.alpha = alpha
        self.beta = beta
        self.pc = pc


    def set_probs(self, pc):
        self.pc = pc

    def simulate(self, query, result_list, dataset):
        clicked_doc = []
        click_label = np.zeros(len(result_list))
        satisfied = True
        for i in range(0, len(result_list)):
            click_prob = np.random.rand()
            docid = result_list[i]

            relevance = dataset.get_relevance_label_by_query_and_docid(query, docid)

            if click_prob <= self.pc[relevance]:
                click_label[i] = 1
                clicked_doc.append(result_list[i])

        return clicked_doc, click_label, satisfied

    def train(self, click_log):
        self._get_train_stat(click_log)

        print("{} training.......".format(self.name))
        for qid in self.stat_dict.keys():
            self.parameter_dict[qid] = {}
            for docID in self.stat_dict[qid].keys():
                a = (self.stat_dict[qid][docID][1] + self.alpha) / (self.stat_dict[qid][docID][0] + self.alpha + self.beta)
                self.parameter_dict[qid][docID] = a

    def _get_train_stat(self, click_log):
        print("{} processing log.......".format(self.name))
        dataset_size = click_log.shape[0]
        for line in range(dataset_size):

            qid = click_log[line][0]
            docIds = click_log[line][1:11]
            clicks = click_log[line][11:21]

            if qid not in self.stat_dict.keys():
                self.stat_dict[qid] = {}

            doc_stat = self.stat_dict[qid]
            for rank in range(10):
                docID = docIds[rank]
                if docID not in doc_stat.keys():
                    doc_stat[docID] = (0, 0)
                exam = doc_stat[docID][0] + 1
                c = doc_stat[docID][1]
                if clicks[rank] == '1':
                    c += 1
                doc_stat[docID] = (exam, c)
            # if line % 10000 == 0:
            #     print("process %d/%d of dataset" % (line, dataset_size))

    def get_click_probs(self, session):
        qid = session[0]
        docIds = session[1:11]
        a_probs = np.zeros(10)
        unseen_docs_index = []
        for i in range(10):
            if docIds[i] not in self.parameter_dict[qid].keys():
                unseen_docs_index.append(i)
                a = self.alpha / (self.alpha + self.beta)
            else:
                a = self.parameter_dict[qid][docIds[i]]
            a_probs[i] = a

        return a_probs

    def get_real_click_probs(self, session, dataset):
        qid = session[0]
        docIds = session[1:11]
        a_probs = np.zeros(10)

        for i in range(10):
            relevance = dataset.get_relevance_label_by_query_and_docid(qid, int(docIds[i]))
            a = self.pc[relevance]
            a_probs[i] = a
        return a_probs
