import numpy as np
import tensorflow as tf


def read_click_log(path):
    print("reading {}.......".format(path))
    if path.split('.')[-1] == "tfrecord":
        log = tf.data.TFRecordDataset(filenames='test.tfrecord')
        print("test")
        return log
    log = []
    num_session = 0
    with open(path) as f:
        for line in f:
            num_session += 1
            cols = line.strip().split()
            log.append(cols)
            # if num_seesion % 10000 == 0:
            #     print("read %d sessions" % (num_seesion))
    f.close()
    print("reading finished, there are %d sessions in the log" % num_session)
    return np.array(log)


def read_query_frequency(path):
    query_freq_dict = {}
    with open(path) as f:
        for line in f:
            cols = line.strip().split()
            freq = cols[0].split(':')[0]
            queries = cols[1:]
            for qid in queries:
                query_freq_dict[qid] = freq
    f.close()
    return query_freq_dict
