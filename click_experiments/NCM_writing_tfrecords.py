import sys
sys.path.append('../')
from clickModel.NCMv2 import NCMv2
from utils import read_file as rf
from utils import utility
from clickModel.SDBN import SDBN
import multiprocessing as mp


def job(click_log_path, output_path, simulator):
    model = NCMv2(64, 10240+1024+1)
    click_log = rf.read_click_log(click_log_path)
    model.initial_representation(click_log)
    model.save_training_set(click_log, output_path, simulator)



if __name__ == "__main__":

    simulators = ["SDBN", "DCTR", "UBM", "Mixed"]

    for r in range(1, 6):
        pool = []
        for simulator in simulators:
            click_log_path = "../click_logs/{}/train_set{}.txt".format(simulator, r)
            output_path = "../click_logs/{}/train_set{}_NCM.tfrecord".format(simulator, r)

            p = mp.Process(target=job, args=(click_log_path, output_path, simulator))
            p.start()
            pool.append(p)

        for p in pool:
                p.join()
        if not utility.send_progress("@arvin writing tfrecord files.".format(simulator), r,
                                     5, "First 5 runs"):
            print("internet disconnect")