import numpy as np
from clickModel.RCM import RCM

class RCTR(RCM):
    def __init__(self, prob=None):
        super().__init__(prob)
        self.name = "RCTR"

    def train(self, click_log):
        print("{} training.......".format(self.name))
        num_clicks = np.zeros(10)
        num_docs = 0
        for session in click_log:
            click_label = session[11:]
            num_docs += 1
            for rank in range(10):
                if click_label[rank] == '1':
                    num_clicks[rank] += 1

        self.prob = num_clicks/num_docs


    def get_click_probs(self, session):
        return np.array(self.prob)

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