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
        self.rank_stat = {}


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
        for i in range(self.iter):
            print(i)
            new_attr_params = copy.deepcopy(self.attr_parameters)
            new_exam_params = copy.deepcopy(self.exam_parameters)

            for qid in self.attr_parameters.keys():
                for docid in self.attr_parameters[qid].keys():

                    # if qid == "4726" and docid == "15":
                    #     print(new_attr_params[qid][docid])
                    #     print(self.query_stat[qid][docid])

                    numerator = 0
                    denominator = 0
                    attr = self.attr_parameters[qid][docid]
                    for rank, click, last_click in self.query_stat[qid][docid]:

                        if click == '1':
                            numerator += 1
                        else:
                            exam = self.exam_parameters[rank][last_click]
                            numerator += ((1 - exam) * attr) / (1 - exam * attr)

                        denominator += 1

                        # if qid == "4726" and docid == "15":
                        #     print(numerator, denominator)


                    new_attr_params[qid][docid] = numerator / denominator

            for rank in self.exam_parameters.keys():
                for last_click in self.exam_parameters[rank].keys():
                    numerator = 0
                    denominator = 0
                    for rank, click, last_click, qid, docID in self.rank_stat[rank][last_click]:
                        if click == '1':
                            numerator += 1
                        else:
                            attr = self.attr_parameters[qid][docID]
                            exam = self.exam_parameters[rank][last_click]
                            numerator += (exam * (1 - attr)) / (1 - exam * attr)
                        denominator += 1

                    new_exam_params[rank][last_click] = numerator / denominator

            self.attr_parameters = new_attr_params
            self.exam_parameters = new_exam_params





    def _init_parameters(self, click_log):
        print("{} processing log.......".format(self.name))
        dataset_size = click_log.shape[0]

        for rank in range(1, 11):
            self.exam_parameters[rank] = {}
            self.rank_stat[rank] = {}
            for i in range(1, rank + 1):
                self.exam_parameters[rank][i] = 0.5
                self.rank_stat[rank][i] = []


        for line in range(dataset_size):
            qid = click_log[line][0]
            docIds = click_log[line][1:11]
            clicks = click_log[line][11:21]
            last_click = 0
            if qid not in self.attr_parameters.keys():
                self.attr_parameters[qid] = {}
                self.query_stat[qid] = {}

            doc_attract = self.attr_parameters[qid]
            doc_stat = self.query_stat[qid]

            for rank in range(len(docIds)):
                docID = docIds[rank]

                self.rank_stat[rank+1][rank+1-last_click].append((rank+1, clicks[rank], rank+1 - last_click, qid, docID))


                if docID not in doc_attract.keys():
                    doc_attract[docID] = 0.2
                    doc_stat[docID] = []

                doc_stat[docID].append((rank+1, clicks[rank], rank+1 - last_click))   # store rank, click, previous click.

                if clicks[rank] == '1':
                    last_click = rank + 1


    def get_click_probs(self, session):
        qid = session[0]
        docIds = session[1:11]

        click_probs = np.zeros(11)
        click_probs[0] = 1
        
        for rank in range(1, 11):  # rank = 2
            click_prob = 0
            for prev_rank in range(rank):  # prev_rank = 0, 1
                no_click_between = 1
                for rank_between in range(prev_rank + 1, rank):  # rank_between = 1
                    if docIds[rank_between-1] not in self.attr_parameters[qid].keys():
                        no_click_between *= (1 - 0.2 *
                                             self.exam_parameters[rank_between][rank_between - prev_rank])
                    else:
                        no_click_between *= (1 - self.attr_parameters[qid][docIds[rank_between-1]] *
                                             self.exam_parameters[rank_between][rank_between - prev_rank])

                if docIds[rank-1] not in self.attr_parameters[qid].keys():
                    click_prob += click_probs[prev_rank] * no_click_between * 0.2 \
                                  * self.exam_parameters[rank][rank - prev_rank]
                else:
                    click_prob += click_probs[prev_rank] * no_click_between * self.attr_parameters[qid][docIds[rank-1]] \
                                 * self.exam_parameters[rank][rank - prev_rank]

                # if click_prob > 1:
                #     print(qid, docIds, rank)

            # make sure every document has chance to be clicked (at lest 1%)

            click_probs[rank] = click_prob
        if np.where(click_probs < 0.01)[0].shape[0] > 0:
            click_probs[np.where(click_probs < 0.01)[0]] = 0.01

        return click_probs[1:]

    def get_real_click_probs(self, session, dataset):
        qid = session[0]
        docIds = session[1:11]

        click_probs = np.zeros(11)
        click_probs[0] = 1

        for rank in range(1, 11):  # rank = 2
            click_prob = 0
            for prev_rank in range(rank):  # prev_rank = 0, 1
                no_click_between = 1
                for rank_between in range(prev_rank + 1, rank):  # rank_between = 1

                    relevance = dataset.get_relevance_label_by_query_and_docid(qid, int(docIds[rank_between - 1]))

                    no_click_between *= (1 - self.pc[relevance] * self.pr[rank_between-1] * self.prr[rank_between - prev_rank -1])

                relevance = dataset.get_relevance_label_by_query_and_docid(qid, int(docIds[rank - 1]))

                click_prob += click_probs[prev_rank] * no_click_between * self.pc[relevance] * \
                              self.pr[rank-1] * self.prr[rank - prev_rank-1]

            click_probs[rank] = click_prob

        return click_probs[1:]
