import tensorflow as tf
from dataset import LetorDataset
import numpy as np
from clickModel.LSTMv2 import LSTMv2
from utils import read_file as rf
from clickModel.DCTR import DCTR
from clickModel.SDBN import SDBN
from utils import utility

# %%

def make_sequence_example(inputs, labels):
    """Returns a SequenceExample for the given inputs and labels.

    Args:
      inputs: A list of input vectors. Each input vector is a list of floats.
      labels: A list of ints.

    Returns:
      A tf.train.SequenceExample containing inputs and labels.
    """
    input_features = [
        tf.train.Feature(float_list=tf.train.FloatList(value=input_))
        for input_ in inputs]
    label_features = [
        tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        for label in labels]
    feature_list = {
        'inputs': tf.train.FeatureList(feature=input_features),
        'labels': tf.train.FeatureList(feature=label_features)
    }
    feature_lists = tf.train.FeatureLists(feature_list=feature_list)
    return tf.train.SequenceExample(feature_lists=feature_lists)

def read_tfrecord(example):
    sequence_features = {
        "inputs": tf.io.FixedLenSequenceFeature([700], dtype=tf.float32),
        "labels": tf.io.FixedLenSequenceFeature([1], dtype=tf.int64)
    }
    # decode the TFRecord
    _, example = tf.io.parse_single_sequence_example(serialized=example, sequence_features=sequence_features)


    return example['inputs'], example['labels']

def session_to_features(session, train_set):
    qid = session[0]
    docids = session[1:11].astype(np.int)
    features = train_set.get_all_features_by_query(qid)[docids]
    return features


def clicks_to_bitmap(clicks):
    clicks = clicks.astype(np.int)
    sess_clicks = clicks.reshape(-1, 1)
    return sess_clicks

train_path = "../datasets/ltrc_yahoo/set1.train.txt"
print("loading training set.......")
train_set = LetorDataset(train_path, 700)

click_log_path = "../feature_click_datasets/Mixed/train_set1.txt"
test_click_log_path =  "../feature_click_datasets/Mixed/seen_set1.txt"
click_log = rf.read_click_log(click_log_path)
test_click_log = rf.read_click_log(test_click_log_path)
query_frequency_path = "../feature_click_datasets/{}/query_frequency{}.txt".format("Mixed", 1)
query_frequency = rf.read_query_frequency(query_frequency_path)
#
# test_logs = {'10': [],
#                  '100': [],
#                  '1000': [],
#                  '10000': [],
#                  '100000': []
#                  }
#
# for i in range(click_log.shape[0]):
#     qid = click_log[i][0]
#     test_logs[query_frequency[qid]].append(click_log[i])
#

writer = tf.io.TFRecordWriter("../feature_click_datasets/Mixed/train_set1.tfrecord")
num_session = 0
for session in click_log:
    inputs = session_to_features(session, train_set)
    labels = clicks_to_bitmap(session[11:21])
    example = make_sequence_example(inputs, labels)
    serialized = example.SerializeToString()
    writer.write(serialized)
    num_session += 1
    if num_session % 1000 == 0:
        print("\r", end='')
        print("num_of_writen:", num_session / 1800000, end="", flush=True)
        if not utility.send_progress("@arvin generate Mixed model .tf file", num_session, 1800000, "train_set1"):
            print("internet disconnect")

writer.close()
print(num_session)
