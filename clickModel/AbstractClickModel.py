

class AbstractClickModel:

    def set_probs(self, pc, ps):
        raise NotImplementedError("Derived class needs to implement "
                                  "set_probs.")

    def simulate(self, query, result_list, dataset):
        raise NotImplementedError("Derived class needs to implement "
                                  "simulate.")

    def train(self, click_log):
        raise NotImplementedError("Derived class needs to implement "
                                  "train.")

    def get_click_probs(self, session):
        raise NotImplementedError("Derived class needs to implement "
                                  "session.")

    def get_perplexity(self, test_click_log):
        raise NotImplementedError("Derived class needs to implement "
                                  "test_click_log.")