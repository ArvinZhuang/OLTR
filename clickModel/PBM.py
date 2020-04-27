import numpy as np
from clickModel.AbstractClickModel import AbstractClickModel


class PBM(AbstractClickModel):
    def __init__(self, pc=None, eta=1):
        self.name = 'PBM'
        self.eta = eta
        self.pc = np.array(pc)

    def simulate(self, query, result_list, dataset):
        propensities = np.power(np.divide(1, np.arange(1.0, len(result_list) + 1)), self.eta)
        rels = np.array(dataset.get_all_relevance_label_by_query(query))
        click_probs = self.pc[rels[result_list]]
        click_probs = click_probs * propensities
        rand = np.random.rand(len(result_list))
        clicks = rand < click_probs
        clicked_doces = result_list[clicks]
        clicks = clicks.astype(int)

        return clicked_doces, clicks, propensities