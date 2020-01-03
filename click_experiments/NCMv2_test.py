import sys
sys.path.append('../')
from clickModel.NCMv2 import NCMv2
from utils import read_file as rf
import numpy as np
import pickle
import bz2
from dataset import LetorDataset
from clickModel.SDBN import SDBN
import sys
import tensorflow as tf


model = NCMv2(64, 10240+1024+1)

pc = [0.05, 0.3, 0.5, 0.7, 0.95]
ps = [0.2, 0.3, 0.5, 0.7, 0.9]
simulator = SDBN(pc, ps)

click_log_path = "../feature_click_datasets/{}/train_set_test.txt".format("SDBN", "1")
# click_log_path = "../click_logs/{}/train_set{}_small.txt".format("SDBN", "1")

click_log = rf.read_click_log(click_log_path)

model.initial_representation(click_log)
# model.save_training_set_numpy(click_log, "test", "SDBN")

# model.save_training_set(click_log, "../click_logs/{}/train_set{}_small_NCM.tfrecord".format("SDBN", "1"), "SDBN")
# model.save_training_set(click_log, "test.tfrecord", "SDBN")
# model.save_training_set_numpy(click_log, "../click_logs/{}/train_set{}_NCM".format("SDBN", "1"), "SDBN")
model.save_training_set_numpy(click_log, "test", "SDBN")

data = np.load("test.npz")
X = data["input"]
Y = data["label"]
#
# data = np.load("../click_logs/{}/train_set{}_NCM.npy.npz".format("SDBN", "1"))
#
# print(data)


# with bz2.BZ2File("../click_logs/{}/train_set{}_NCM_input.txt".format("SDBN", "1"), 'rb') as fp:
#     X = pickle.load(fp)
#
# with bz2.BZ2File("../click_logs/{}/train_set{}_NCM_label.txt".format("SDBN", "1"), 'rb') as fp:
#     Y = pickle.load(fp)
model.train_tfrecord("test.tfrecord".format("SDBN", "1"), 774, 100)
# model.train(X, Y)
#
#
# model.train_tfrecord("../click_logs/{}/train_set{}_small_NCM.tfrecord".format("SDBN", "1"), 30, 100)
#

pc = [0.05, 0.3, 0.5, 0.7, 0.95]
ps = [0.2, 0.3, 0.5, 0.7, 0.9]
base_line = SDBN(pc, ps)
base_line.train(click_log)

print(click_log[0])
print(base_line.get_click_probs(click_log[0]))
print(model.get_click_probs(click_log[0], X[0]))
