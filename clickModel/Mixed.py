from clickModel.SDBN import SDBN
from clickModel.SDCM import SDCM
from clickModel.CM import CM
from clickModel.DCTR import DCTR
from clickModel.UBM import UBM
from clickModel.AbstractClickModel import AbstractClickModel

import numpy as np

class Mixed(AbstractClickModel):
    def __init__(self, models):
        self.models = models
        self.name = 'Mixed'

    def simulate(self, query, result_list, dataset):
        simolator = np.random.choice(self.models)
        session = [query]
        session.extend(result_list)
        clicked_doc, click_label, satisfied = simolator.simulate(query, result_list, dataset)
        return clicked_doc, click_label, satisfied, simolator.name


