import numpy as np
import tensorflow as tf
from dataset import LetorDataset
import numpy as np
from clickModel.LSTMv2 import LSTMv2
from utils import read_file as rf
from clickModel.DCTR import DCTR

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

train_path = "../datasets/ltrc_yahoo/test_set.txt"
print("loading training set.......")
train_set = LetorDataset(train_path, 700)

click_log_path = "../datasets/ltrc_yahoo/test_click_log.txt"
test_click_log_path = "../datasets/ltrc_yahoo/test_click_log_test.txt"
click_log = rf.read_click_log(click_log_path)
test_click_log = rf.read_click_log(test_click_log_path)




writer = tf.io.TFRecordWriter('test.tfrecord')
for seesion in click_log:
    inputs = session_to_features(seesion, train_set)
    labels = clicks_to_bitmap(seesion[11:])
    example = make_sequence_example(inputs, labels)
    serialized = example.SerializeToString()
    writer.write(serialized)  # **4.写入文件中
writer.close()



dataset = tf.data.TFRecordDataset(filenames = 'test.tfrecord')
dataset = dataset.map(read_tfrecord)
dataset = dataset.repeat(1)
dataset = dataset.shuffle(2048)
dataset = dataset.batch(32, drop_remainder=False)

#%%
pc = [0.05, 0.3, 0.5, 0.7, 0.95]
ps = [0.2, 0.3, 0.5, 0.7, 0.9]
simulator = DCTR(pc)
print(click_log.shape)
print(test_click_log.shape)
#
click_model = LSTMv2(700, 1024, train_set)
print(click_model.get_MSE(test_click_log[np.random.choice(test_click_log.shape[0], 100)], train_set, simulator))
click_model.train(None, dataset, 10)
print(click_model.get_MSE(test_click_log[np.random.choice(test_click_log.shape[0], 100)], train_set, simulator))