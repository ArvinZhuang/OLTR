import numpy as np
from clickModel.AbstractClickModel import AbstractClickModel

class RCM(AbstractClickModel):
    def __init__(self, prob):
        self.name = "RCM"
        self.prob = prob



    def get_click_probs(self, session):
        return np.ones(10) * self.prob

    def get_perplexity(self, test_click_log):
        print(self.name, "computing perplexity")
        perplexity = np.zeros(10)
        size = test_click_log.shape[0]
        for i in range(size):
            session = test_click_log[i][:11]
            click_label = test_click_log[i][11:]
            click_probs = self.get_click_probs(session)
            for rank, click_prob in enumerate(click_probs):
                if click_label[rank] == '1':
                    p = click_prob
                else:
                    p = 1 - click_prob

                with np.errstate(invalid='raise'):
                    try:
                        p = 0.001 if p < 0.001 else p
                        perplexity[rank] += np.log2(p)
                    except:
                        print("error!, p=", p)
                        print(session, rank + 1)
                        perplexity[rank] += 0

        perplexity = [2 ** (-x / size) for x in perplexity]
        return perplexity

    def get_MSE(self, test_click_log, dataset, simulator):
        print(self.name, "computing MSE")
        MSE = np.zeros(10)
        size = test_click_log.shape[0]
        for i in range(size):
            session = test_click_log[i]
            click_probs = self.get_click_probs(session)
            real_click_probs = simulator.get_real_click_probs(session, dataset)
            MSE += np.square(click_probs - real_click_probs)

        return MSE/size