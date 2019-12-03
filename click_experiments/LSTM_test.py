from clickModel.SDBN import SDBN
from clickModel.SDCM import SDCM
from clickModel.CM import CM
from clickModel.DCTR import DCTR
from clickModel.LSTM import LSTM
from clickModel.LSTMv2 import LSTMv2
from utils import read_file as rf
from dataset import LetorDataset
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp



def run(click_log, test_click_log, query_frequency, click_model, train_set, simulator, f):

    test_logs = {'10': [],
                 '100': [],
                 '1000': [],
                 '10000': [],
                 '100000': []
                 }

    for i in range(test_click_log.shape[0]):
        qid = test_click_log[i][0]
        test_logs[query_frequency[qid]].append(test_click_log[i])



    num_trained = 0
    for epoc in range(100):
        for subLog in np.split(click_log, 10000):
            print("\r", end='')
            print("epoc:", epoc, "num_of_trained:", num_trained/1800000, end="", flush=True)
            click_model.train(subLog)
            num_trained += subLog.shape[0]
        print(click_model.get_MSE(test_click_log[np.random.choice(test_click_log.shape[0], 1000)],
                                  train_set,
                                  simulator))
            # print(click_model.get_MSE(test_click_log, train_set,
            #                           simulator))

    # frequencies = ['10', '100', '1000', '10000', '100000']
    # # i = 0
    #
    # f.write("Click Model:" + cm.name + "\n")
    #
    # for freq in frequencies:
    #     perplexities = click_model.get_perplexity(np.array(test_logs[freq]))
    #     MSEs = click_model.get_MSE(np.array(test_logs[freq]), train_set, simulator)
    #
    #     perplexity_line = "Frequency " + freq + " perplexities:"
    #     MSEs_line = "Frequency " + freq + " MSE:"
    #     for perp in perplexities:
    #         perplexity_line += " " + str(perp)
    #     for MSE in MSEs:
    #         MSEs_line += " " + str(MSE)
    #     f.write(perplexity_line + "\n")
    #     f.write(MSEs_line + "\n")
    #
    # f.close()
# %%
train_path = "../datasets/ltrc_yahoo/set1.train.txt"
# train_path = "../datasets/ltrc_yahoo/test_set.txt"
test_path = "../datasets/ltrc_yahoo/set1.test.txt"
print("loading training set.......")
train_set = LetorDataset(train_path, 700)
# %%
# print("loading testing set.......")
# test_set = LetorDataset(test_path, 700)
pc = [0.05, 0.3, 0.5, 0.7, 0.95]
ps = [0.2, 0.3, 0.5, 0.7, 0.9]

dataset = 'SDBN'
simulator = SDBN(pc, ps)
id = 1

click_log_path = "../feature_click_datasets/{}/train_set{}.txt".format(dataset, id)
test_click_log_path = "../feature_click_datasets/{}/seen_set{}.txt".format(dataset, id)
query_frequency_path = "../feature_click_datasets/{}/query_frequency{}.txt".format(dataset, id)
# click_log_path = "../datasets/ltrc_yahoo/test_click_log.txt"
# test_click_log_path = "../datasets/ltrc_yahoo/test_click_log_test.txt"
click_log = rf.read_click_log(click_log_path)
test_click_log = rf.read_click_log(test_click_log_path)
query_frequency = rf.read_query_frequency(query_frequency_path)

cm = LSTMv2(700, 1024, train_set)


f = open("../click_model_results/{}/seen_set{}_{}_result.txt".format(dataset, id, cm.name)
                         , "w+")
# %%
run(click_log, test_click_log, query_frequency, cm, train_set, simulator, f)

