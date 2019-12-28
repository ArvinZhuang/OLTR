from clickModel.SDBN import SDBN
from clickModel.SDCM import SDCM
from clickModel.CM import CM
from clickModel.DCTR import DCTR
from clickModel.UBM import UBM
from clickModel.AbstractClickModel import AbstractClickModel

import numpy as np

class Mixed(CM):
    def __init__(self, models):
        self.models = models
        self.name = 'Mixed'
        self.model_names = []
        for model in models:
            self.model_names.append(model.name)

    def simulate(self, query, result_list, dataset):
        simolator = np.random.choice(self.models)
        session = [query]
        session.extend(result_list)
        clicked_doc, click_label, satisfied = simolator.simulate(query, result_list, dataset)
        return clicked_doc, click_label, satisfied, simolator.name

    def get_real_click_probs(self, session, dataset):
        simulator_name = session[-1]
        model_index = self.model_names.index(simulator_name)
        return self.models[model_index].get_real_click_probs(session, dataset)



        # click_probs = np.zeros(10)
        # for simulator in self.models:
        #     click_probs += simulator.get_real_click_probs(session, dataset)
        # return click_probs/len(self.models)


