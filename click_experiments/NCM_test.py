from clickModel.NCM_TF import NCM
from utils import read_file as rf
import numpy as np
import pickle
from dataset import LetorDataset
from clickModel.SDBN import SDBN

model = NCM(774, 100, 10240+1024+1, 2)

pc = [0.05, 0.3, 0.5, 0.7, 0.95]
ps = [0.2, 0.3, 0.5, 0.7, 0.9]
simulator = SDBN(pc, ps)

click_log_path = "../feature_click_datasets/{}/train_set{}.txt".format("SDBN", "_test")

click_log = rf.read_click_log(click_log_path)

model.initial_representation(click_log)

# session = np.array(['1112', '16', '3', '45', '37', '31', '22', '5', '34', '17', '21', '0', '0', '1', '0', '0', '0', '0', '0', '0', '0' ])
#
model.save_training_set(click_log, "")

# with open("X.txt", "rb") as fp:
#     X = pickle.load(fp)
#
# with open("Y.txt", "rb") as fp:
#     Y = pickle.load(fp)
#
# train_path = "../datasets/ltrc_yahoo/set1.train.txt"
#
# print("loading training set.......")
# train_set = LetorDataset(train_path, 700)
# model.train(X, Y)
# # model.get_MSE(click_log, train_set, simulator)
