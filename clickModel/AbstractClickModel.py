

class AbstractClickModel:
    def __init__(self, dataset, pc, ps):
        self.dataset = dataset
        self.pc = pc
        self.ps = ps

    def set_probs(self, pc, ps):
        raise NotImplementedError("Derived class needs to implement "
                                  "set_probs.")

    def simulate(self, query, result_list):
        raise NotImplementedError("Derived class needs to implement "
                                  "simulate.")

