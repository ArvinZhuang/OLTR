import sys
sys.path.append('../')
from clickModel.SDBN import SDBN
from clickModel.SDBN_reverse import SDBN_reverse
from clickModel.SDCM import SDCM
from clickModel.CM import CM
from clickModel.DCTR import DCTR
from clickModel.UBM import UBM
from clickModel.RCM import RCM
from clickModel.RCTR import RCTR
from clickModel.Mixed import Mixed
from utils import read_file as rf
from utils import utility
from dataset import LetorDataset
# import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp


train_path = "../datasets/ltrc_yahoo/set1.train.txt"
print("loading training set.......")
train_set = LetorDataset(train_path, 700)

pc = [0.05, 0.3, 0.5, 0.7, 0.95]
ps = [0.2, 0.3, 0.5, 0.7, 0.9]
mixed_models = [DCTR(pc), SDBN(pc, ps), UBM(pc)]
datasets_simulator = [
                    ('SDBN', SDBN(pc, ps)),
                      # ('SDCM', SDCM(pc)),
                      # ('CM', CM(pc)),
                      ('DCTR', DCTR(pc)),
                      ('UBM', UBM(pc)),
                    ('SDBN_reverse', SDBN_reverse(pc, ps))
                        ]
click_model = RCTR()

for dataset, simulator in datasets_simulator:
    for id in range(1, 16):
        click_log_path = "../click_logs/{}/train_set{}.txt".format(dataset, id)
        click_log = rf.read_click_log(click_log_path)
        click_model.train(click_log)

        test_click_log_path = "../click_logs/{}/unseen_set{}.txt".format(dataset, id)
        query_frequency_path = "../click_logs/{}/query_frequency{}.txt".format(dataset, id)
        test_click_log = rf.read_click_log(test_click_log_path)
        query_frequency = rf.read_query_frequency(query_frequency_path)



        f = open("../click_model_results/{}/unseen_set{}_{}_result.txt".format(dataset, id, click_model.name)
                 , "w+")
        f.write("Click Model:" + click_model.name + "\n")

        perplexities = click_model.get_perplexity(np.array(test_click_log))
        MSEs = click_model.get_MSE(np.array(test_click_log), train_set, simulator)

        perplexity_line = "perplexities:"
        MSEs_line = "MSE:"
        for perp in perplexities:
            perplexity_line += " " + str(perp)
        for MSE in MSEs:
            MSEs_line += " " + str(MSE)
        f.write(perplexity_line + "\n")
        f.write(MSEs_line + "\n")

        f.close()
