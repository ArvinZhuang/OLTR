import numpy as np
from clickModel.AbstractClickModel import AbstractClickModel


class CascadeClickModel(AbstractClickModel):
    def __init__(self, dataset, pc, ps):
        super().__init__(dataset, pc, ps)

    def set_probs(self, pc, ps):
        self.pc = pc
        self.ps = ps

    def simulate(self, query, result_list):
        clicked_doc = []
        click_label = np.zeros(len(result_list))

        for i in range(0, len(result_list)):
            click_prob = np.random.rand()
            stop_prob = np.random.rand()
            docid = result_list[i]

            relevance = self.dataset.get_relevance_label_by_query_and_docid(query, docid)

            if click_prob <= self.pc[relevance]:
                click_label[i] = 1
                clicked_doc.append(result_list[i])
                if stop_prob <= self.ps[relevance]:
                    break
        return clicked_doc, click_label
