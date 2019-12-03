from dataset import LetorDataset
import numpy as np
from clickModel.LSTMv2 import LSTMv2
from utils import read_file as rf
from clickModel.DCTR import DCTR

train_path = "../datasets/ltrc_yahoo/test_set.txt"
print("loading training set.......")
train_set = LetorDataset(train_path, 700)

click_log_path = "../datasets/ltrc_yahoo/test_click_log.txt"
test_click_log_path = "../datasets/ltrc_yahoo/test_click_log_test.txt"
click_log = rf.read_click_log(click_log_path)
test_click_log = rf.read_click_log(test_click_log_path)

pc = [0.05, 0.3, 0.5, 0.7, 0.95]
ps = [0.2, 0.3, 0.5, 0.7, 0.9]
simulator = DCTR(pc)
print(click_log.shape)
print(test_click_log.shape)
#
click_model = LSTMv2(700, 1024, train_set)
click_model.train(click_log)
print(click_model.get_MSE(test_click_log[np.random.choice(test_click_log.shape[0], 100)], train_set, simulator))