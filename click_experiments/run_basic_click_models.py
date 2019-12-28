from clickModel.SDBN import SDBN
from clickModel.SDCM import SDCM
from clickModel.CM import CM
from clickModel.DCTR import DCTR
from clickModel.UBM import UBM
from clickModel.Mixed import Mixed
from utils import read_file as rf
from utils import utility
from dataset import LetorDataset
# import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp

COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k']


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

    click_model.train(click_log)

    frequencies = ['10', '100', '1000', '10000', '100000']
    # i = 0

    f.write("Click Model:" + cm.name + "\n")

    for freq in frequencies:
        perplexities = click_model.get_perplexity(np.array(test_logs[freq]))
        MSEs = click_model.get_MSE(np.array(test_logs[freq]), train_set, simulator)

        perplexity_line = "Frequency " + freq + " perplexities:"
        MSEs_line = "Frequency " + freq + " MSE:"
        for perp in perplexities:
            perplexity_line += " " + str(perp)
        for MSE in MSEs:
            MSEs_line += " " + str(MSE)
        f.write(perplexity_line + "\n")
        f.write(MSEs_line + "\n")

    f.close()

        # plt.plot(range(10), perplexity, color=COLORS[i], alpha=1)
        # print("Average Perplexity for freq {}: {}".format(freq, np.sum(perplexity)))
        # print("Average MSE for freq {}: {}".format(freq, np.sum(MSE)))
        # i += 1
    # plt.figure(1)
    # plt.legend(frequencies, loc='lower left')
    # plt.show()

# %%
if __name__ == "__main__":
    # %%
    train_path = "../datasets/ltrc_yahoo/set1.train.txt"
    test_path = "../datasets/ltrc_yahoo/set1.test.txt"
    print("loading training set.......")
    train_set = LetorDataset(train_path, 700)
    # %%
    # print("loading testing set.......")
    # test_set = LetorDataset(test_path, 700)
    pc = [0.05, 0.3, 0.5, 0.7, 0.95]
    ps = [0.2, 0.3, 0.5, 0.7, 0.9]
    mixed_models = [DCTR(pc), CM(pc), SDBN(pc, ps), SDCM(pc), UBM(pc)]
    datasets_simulator = [
                        # ('SDBN', SDBN(pc, ps)),
                          # ('SDCM', SDCM(pc)),
                          # ('CM', CM(pc)),
                          # ('DCTR', DCTR(pc)),
                          # ('UBM', UBM(pc)),
                        ('Mixed', Mixed(mixed_models))]
    # datasets = ['CM']
    progress = 0
    for dataset, simulator in datasets_simulator:
        for id in range(1, 12):
            click_log_path = "../feature_click_datasets/{}/train_set{}.txt".format(dataset, id)
            test_click_log_path = "../feature_click_datasets/{}/seen_set{}.txt".format(dataset, id)
            query_frequency_path = "../feature_click_datasets/{}/query_frequency{}.txt".format(dataset, id)
            click_log = rf.read_click_log(click_log_path)
            test_click_log = rf.read_click_log(test_click_log_path)
            query_frequency = rf.read_query_frequency(query_frequency_path)

            click_models = [SDBN(),
                            SDCM(),
                            CM(),
                            DCTR(),
                            UBM()]

            processors = []
            for cm in click_models:
                print(dataset, cm.name, "running!")
                f = open("../click_model_results/{}/seen_set{}_{}_result.txt".format(dataset, id, cm.name)
                         , "w+")
                p = mp.Process(target=run,
                           args=(click_log, test_click_log, query_frequency, cm, train_set, simulator, f))
                p.start()
                processors.append(p)

            for p in processors:
                p.join()

            progress += 1

            if not utility.send_progress("Basic click model experiments", progress, 10, "{} run {}".format(dataset, id)):
                print("internet disconnect")




