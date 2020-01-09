import sys
sys.path.append('../')
from clickModel.DFBNCM import DFBNCM
from utils import read_file as rf
from utils import utility
from clickModel.SDBN import SDBN
import multiprocessing as mp
from dataset import LetorDataset


def job(click_log_path, output_path, simulator, dataset):
    model = DFBNCM(64, 1024, 10240, 700, 700, dataset)
    click_log = rf.read_click_log(click_log_path)
    model.initial_representation(click_log)
    model.save_training_tfrecord(click_log, output_path, simulator)



if __name__ == "__main__":

    simulators = ["SDBN", "DCTR", "UBM"]

    dataset_path = "../datasets/ltrc_yahoo/set1.train.txt"
    print("loading training set.......")
    dataset = LetorDataset(dataset_path, 700)
    for r in range(3, 16):
        pool = []
        for simulator in simulators:
            click_log_path = "../click_logs/{}/train_set{}.txt".format(simulator, r)
            output_path = "../click_logs/{}/train_set{}_DFBNCM.tfrecord".format(simulator, r)

            p = mp.Process(target=job, args=(click_log_path, output_path, simulator, dataset))
            p.start()
            pool.append(p)

        for p in pool:
                p.join()
