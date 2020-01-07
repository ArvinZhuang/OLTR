import sys
sys.path.append('../')
import tensorflow as tf
from dataset import LetorDataset
import numpy as np
from clickModel.LSTMv2 import LSTMv2
from utils import read_file as rf
from clickModel.SDBN import SDBN
from clickModel.SDCM import SDCM
from clickModel.CM import CM
from clickModel.DCTR import DCTR
from clickModel.UBM import UBM
from clickModel.Mixed import Mixed
from utils import read_file as rf
from tensorflow.keras.models import load_model


train_path = "../datasets/ltrc_yahoo/set1.train.txt"
print("loading training set.......")
train_set = LetorDataset(train_path, 700)

generator = "Mixed"

click_log_path = "../feature_click_datasets/{}/train_set1.txt".format(generator)
test_click_log_path =  "../feature_click_datasets/{}/seen_set1.txt".format(generator)
click_log = rf.read_click_log(click_log_path)
test_click_log = rf.read_click_log(test_click_log_path)


# #
dataset = tf.data.TFRecordDataset(filenames='../feature_click_datasets/{}/train_set1.tfrecord'.format(generator))
# # # test_dataset = tf.data.TFRecordDataset(filenames='../feature_click_datasets/SDBN/seen_set1.tfrecord')
# # #%%
pc = [0.05, 0.3, 0.5, 0.7, 0.95]
ps = [0.2, 0.3, 0.5, 0.7, 0.9]
Mixed_models = [DCTR(pc), CM(pc), SDBN(pc, ps), SDCM(pc), UBM(pc)]
simulator = Mixed(Mixed_models)
print(click_log.shape)
print(test_click_log.shape)
#
click_model = LSTMv2(700, 1024, train_set, batch_size=128, epoch=5)
print(click_model.get_MSE(test_click_log[np.random.choice(test_click_log.shape[0], 1000)], train_set, simulator))
click_model.train(dataset)

print(click_model.get_MSE(test_click_log[np.random.choice(test_click_log.shape[0], 1000)], train_set, simulator))

click_model.model.save("../click_model_results/LSTM_models/{}_train_set1.h5".format(generator))



# test model
# train_path = "../datasets/ltrc_yahoo/set1.train.txt"
# test_path = "../datasets/ltrc_yahoo/set1.test.txt"
# print("loading training set.......")
# train_set = LetorDataset(train_path, 700)

# click_log_path = "../feature_click_datasets/{}/train_set{}.txt".format("SDBN", 1)
test_click_log_path = "../feature_click_datasets/{}/seen_set{}.txt".format(generator, 1)
query_frequency_path = "../feature_click_datasets/{}/query_frequency{}.txt".format(generator, 1)
# click_log = rf.read_click_log(click_log_path)
test_click_log = rf.read_click_log(test_click_log_path)
query_frequency = rf.read_query_frequency(query_frequency_path)

f = open("../click_model_results/{}/seen_set{}_{}_result.txt".format(generator, 1, "LSTM")
                         , "w+")

test_logs = {'10': [],
                 '100': [],
                 '1000': [],
                 '10000': [],
                 '100000': []
                 }

for i in range(test_click_log.shape[0]):
    qid = test_click_log[i][0]
    test_logs[query_frequency[qid]].append(test_click_log[i])

click_model = LSTMv2(700, 1024, train_set, model=load_model('../click_model_results/LSTM_models/{}_train_set1.h5'.format(generator)))
# print(click_model.get_MSE(np.array(test_logs["10"]), train_set, simulator))

frequencies = ['10', '100', '1000', '10000', '100000']
# i = 0

f.write("Click Model:" + "LSTM" + "\n")

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