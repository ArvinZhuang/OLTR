# %%
import sys
sys.path.append('../')
from dataset.LetorDataset import LetorDataset
import numpy as np
import multiprocessing as mp
import random
from ranker.PDGDLinearRanker import PDGDLinearRanker
from clickModel.SDCM import SDCM
from clickModel.CM import CM
from clickModel.DCTR import DCTR
from clickModel.SDBN import SDBN
from clickModel.SDBN_reverse import SDBN_reverse
from clickModel.UBM import UBM
from clickModel.Mixed import Mixed
from utils import evl_tool
from utils import utility


def get_experimental_queries(train_set, out_path, id):
    train_queries = train_set.get_all_querys()

    clean_train_queries = []
    for q in train_queries:
        if len(train_set.get_candidate_docids_by_query(q)) >= 10:
            clean_train_queries.append(q)

    random.shuffle(clean_train_queries)

    train_query_stream = []
    seen_query_stream = []
    unseen_query_stream = []

    query_set_size = [10000, 1000, 100, 10]
    frequency = 10

    f = open(out_path + "query_frequency%d.txt" % (id), "w+")
    for l in range(len(query_set_size)):
        line = "%d: " % (frequency)
        for i in range(query_set_size[l]):
            query = [clean_train_queries.pop(0)]
            train_query_stream.extend(query * frequency)
            seen_query_stream.extend(query * int(frequency/10))
            line += query[0] + " "
        f.write(line + "\n")
        frequency *= 10
    f.close()
    random.shuffle(train_query_stream)

    clean_unseen_queries = clean_train_queries

    for query in clean_unseen_queries:
        unseen_query_stream.extend([query] * 10)

    return train_query_stream, seen_query_stream, unseen_query_stream


def generate_dataset(train_set, cm, out_path, id):
    print("simulating", cm.name, "click log", id)
    train_queries, seen_queries, unseen_queries = get_experimental_queries(train_set, out_path, id)

    ranker = PDGDLinearRanker(700, 0.1, 1)

    f = open(out_path + "train_set%d.txt" % (id), "w+")
    num_queries = len(train_queries)
    index = 0
    while index < num_queries:
        qid = train_queries[index]
        result_list, scores = ranker.get_query_result_list(train_set, qid)
        if cm.name == "Mixed":
            clicked_doc, click_label, satisfied, s_name = cm.simulate(qid, result_list, train_set)
        else:
            clicked_doc, click_label, satisfied = cm.simulate(qid, result_list, train_set)
        if not satisfied:
            continue

        ranker.update_to_clicks(click_label, result_list, scores, train_set.get_all_features_by_query(qid))
        index += 1
        line = qid + " "
        for d in result_list.tolist():
            line += str(d) + " "
        for c in click_label.tolist():
            line += str(int(c)) + " "
        if cm.name == "Mixed":
            line += s_name + " "
        line += "\n"
        f.write(line)
        # if index % 10000 == 0:
        # print("write %d/%d queries" % (index, num_queries))
    f.close()
    print(cm.name, "train_set finished!")

    f = open(out_path + "seen_set%d.txt" % (id), "w+")
    num_queries = len(seen_queries)
    index = 0
    while index < num_queries:
        qid = seen_queries[index]
        result_list, scores = ranker.get_query_result_list(train_set, qid)
        if cm.name == "Mixed":
            clicked_doc, click_label, satisfied, s_name = cm.simulate(qid, result_list, train_set)
        else:
            clicked_doc, click_label, satisfied = cm.simulate(qid, result_list, train_set)
        if not satisfied:
            continue
        index += 1
        line = qid + " "
        for d in result_list.tolist():
            line += str(d) + " "
        for c in click_label.tolist():
            line += str(int(c)) + " "
        if cm.name == "Mixed":
            line += s_name + " "
        line += "\n"
        f.write(line)
        # if index % 10000 == 0:
        # print("write %d/%d queries" % (index, num_queries))
    f.close()
    print(cm.name, "seen_set finished!")

    f = open(out_path + "unseen_set%d.txt" % (id), "w+")
    num_queries = len(unseen_queries)
    index = 0
    while index < num_queries:
        qid = unseen_queries[index]
        result_list, scores = ranker.get_query_result_list(train_set, qid)
        if cm.name == "Mixed":
            clicked_doc, click_label, satisfied, s_name = cm.simulate(qid, result_list, train_set)
        else:
            clicked_doc, click_label, satisfied = cm.simulate(qid, result_list, train_set)
        if not satisfied:
            continue
        index += 1
        line = qid + " "
        for d in result_list.tolist():
            line += str(d) + " "
        for c in click_label.tolist():
            line += str(int(c)) + " "
        if cm.name == "Mixed":
            line += s_name + " "
        line += "\n"
        f.write(line)
        # if index % 10000 == 0:
        # print("write %d/%d queries" % (index, num_queries))
    f.close()
    print(cm.name, "unseen_set finished!")


# %%
if __name__ == "__main__":
    # %%
    train_path = "../datasets/ltrc_yahoo/set1.train.txt"
    print("loading training set.......")
    train_set = LetorDataset(train_path, 700)

    # %%
    # pc = [0.4, 0.6, 0.7, 0.8, 0.9]
    # ps = [0.1, 0.2, 0.3, 0.4, 0.5]
    pc = [0.05, 0.3, 0.5, 0.7, 0.95]
    ps = [0.2, 0.3, 0.5, 0.7, 0.9]

    # click_models = [DCTR(pc), SDBN(pc, ps), UBM(pc), SDBN_reverse(pc, ps)]
    # Mixed_model = Mixed(click_models)
    # simulators = [DCTR(pc), SDBN(pc, ps), UBM(pc), SDBN_reverse(pc, ps), Mixed_model]
    simulators = [SDBN_reverse(pc, ps)]

    for id in range(2, 16):
        pool = []
        for cm in simulators:
            p = mp.Process(target=generate_dataset,
                        args=(train_set, cm, "../click_logs/{}/".format(cm.name), id))
            p.start()
            pool.append(p)
        for p in pool:
            p.join()
            # utility.send_progress("@arvin generating "+cm.name + " click log.",
            #                   id, 15, "generating smaller click log.")