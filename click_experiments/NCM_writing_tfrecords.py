from clickModel.NCMv2 import NCMv2
from utils import read_file as rf

from clickModel.SDBN import SDBN
import multiprocessing as mp

def job(click_log_path, output_path, simulator):

    model = NCMv2(64, 10240+1024+1)

    click_log_path = click_log_path

    click_log = rf.read_click_log(click_log_path)

    model.initial_representation(click_log)

    model.save_training_set(click_log, output_path, simulator)



if __name__ == "__main__":

    simulators = ["SDBN", "Mixed"]

    for simulator in simulators:

        click_log_path = "../feature_click_datasets/{}/train_set{}.txt".format(simulator, "1")
        output_path = "../feature_click_datasets/{}/train_set{}_NCM.tfrecord".format(simulator, "1")

        p = mp.Process(target=job, args=(click_log_path, output_path, simulator))
        p.start()