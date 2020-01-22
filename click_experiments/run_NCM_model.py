import sys
sys.path.append('../')
import tensorflow as tf
from dataset import LetorDataset
import numpy as np
from utils import read_file as rf
from clickModel.SDBN import SDBN
from clickModel.SDBN_reverse import SDBN_reverse
from clickModel.SDCM import SDCM
from clickModel.CM import CM
from clickModel.DCTR import DCTR
from clickModel.UBM import UBM
from clickModel.Mixed import Mixed
from clickModel.NCM import NCM
from keras.models import load_model



def run(simulator, dataset, run):

    click_model = NCM(256, 1024, 10240)
                      # model=load_model('../click_model_results/NCM_model/{}/train_set{}.h5'.format(simulator.name, run)))

    click_log_path = "../click_logs/{}/train_set{}.txt".format(simulator.name, run)
    click_log = rf.read_click_log(click_log_path)
    click_model.initial_representation(click_log)

    click_model.train_tfrecord('../click_logs/{}/train_set{}_NCM.tfrecord'.format(simulator.name, run),
                               batch_size=64,
                               epoch=50,
                               steps_per_epoch=1)

    click_model.model.save("../click_model_results/NCM_model/{}/train_set{}.h5".format(simulator.name, run))


    test_click_log_path = "../click_logs/{}/seen_set{}.txt".format(simulator.name, run)
    query_frequency_path = "../click_logs/{}/query_frequency{}.txt".format(simulator.name, run)
    test_click_log = rf.read_click_log(test_click_log_path)
    query_frequency = rf.read_query_frequency(query_frequency_path)

    f = open("../click_model_results/{}/seen_set{}_{}_result.txt".format(simulator.name, run, "NCM")
                             , "w+")

    test_logs = {'10': [],
                     '100': [],
                     '1000': [],
                     '10000': []
                     }

    for i in range(test_click_log.shape[0]):
        qid = test_click_log[i][0]
        test_logs[query_frequency[qid]].append(test_click_log[i])

    frequencies = ['10', '100', '1000', '10000']
    # i = 0

    f.write("Click Model:" + "NCM" + "\n")

    for freq in frequencies:
        perplexities = click_model.get_perplexity(np.array(test_logs[freq]))
        MSEs = click_model.get_MSE(np.array(test_logs[freq]), dataset, simulator)

        perplexity_line = "Frequency " + freq + " perplexities:"
        MSEs_line = "Frequency " + freq + " MSE:"
        for perp in perplexities:
            perplexity_line += " " + str(perp)
        for MSE in MSEs:
            MSEs_line += " " + str(MSE)
        f.write(perplexity_line + "\n")
        f.write(MSEs_line + "\n")

    f.close()

if __name__ == "__main__":
    pc = [0.05, 0.3, 0.5, 0.7, 0.95]
    ps = [0.2, 0.3, 0.5, 0.7, 0.9]
    # Mixed_models = [DCTR(pc), SDBN(pc, ps), UBM(pc)]
    # simulators = [SDBN(pc, ps), Mixed(Mixed_models), DCTR(pc), UBM(pc)]
    # simulators = [DCTR(pc)]
    simulators = [UBM(pc), SDBN_reverse(pc, ps)]

    dataset_path = "../datasets/ltrc_yahoo/set1.train.txt"
    print("loading training set.......")
    dataset = LetorDataset(dataset_path, 700)

    for r in range(10, 11):
        for simulator in simulators:
            run(simulator, dataset, r)
