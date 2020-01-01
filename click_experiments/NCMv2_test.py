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

click_log_path = "../feature_click_datasets/{}/train_set{}.txt".format("SDBN", "1")

click_log = rf.read_click_log(click_log_path)

model.initial_representation(click_log)

model.save_training_set(click_log, "../feature_click_datasets/{}/train_set{}_NCM.tfrecord".format("SDBN", "1"))


# model.train_tfrecord('test.tfrecord', 774, 20)

# session = np.array(['1112', '16', '3', '45', '37', '31', '22', '5', '34', '17', '21', '0', '0', '1', '0', '0', '0', '0', '0', '0', '0' ])
#
# model.save_training_set(click_log, "")
#
# with bz2.BZ2File("Xv2.txt", 'rb') as fp:
#     X = pickle.load(fp)
#
# with bz2.BZ2File("Yv2.txt", 'rb') as fp:
#     Y = pickle.load(fp)
# print(X.shape)
# print(Y.shape)
# print(X[0])
#
# # model.train(X, Y)
# for session in click_log:
#     model.predict(session)
#
# pc = [0.05, 0.3, 0.5, 0.7, 0.95]
# ps = [0.2, 0.3, 0.5, 0.7, 0.9]
# base_line = SDBN(pc, ps)
# base_line.train(click_log)
# print(base_line.get_click_probs(click_log[0]))