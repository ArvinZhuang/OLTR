import sys
sys.path.append('../')
from clickModel.NCM import NCM
from utils import read_file as rf
from utils import utility
from clickModel.SDBN import SDBN
import multiprocessing as mp


def job(click_log_path, output_path, simulator):
    model = NCM(64, 1024, 10240)
    click_log = rf.read_click_log(click_log_path)
    model.initial_representation(click_log)
    model.save_training_tfrecord(click_log, output_path, simulator)



if __name__ == "__main__":

    simulators = ["SDBN", "DCTR", "UBM", "SDBN_reverse"]

    for r in range(1, 16):
        pool = []
        for simulator in simulators:
            click_log_path = "../click_logs/{}/train_set{}.txt".format(simulator, r)
            output_path = "../click_logs/{}/train_set{}_NCM.tfrecord".format(simulator, r)

            p = mp.Process(target=job, args=(click_log_path, output_path, simulator))
            p.start()
            pool.append(p)

        for p in pool:
                p.join()
