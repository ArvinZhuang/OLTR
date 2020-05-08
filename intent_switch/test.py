import os
import numpy as np
from dataset import LetorDataset


def read_intent_qrel(path: str):

    # q-d pair dictionary
    query_dic = {}

    with open(path, 'r') as f:
        for line in f:
            qid, _, docid, rel = line.strip().split()
            if qid in query_dic.keys():
                query_dic[qid][docid] = int(rel)
            else:
                query_dic[qid] = {docid: int(rel)}
    return query_dic

def get_dataset_avg_num_of_rel(dataset):
    avg = 0
    for qid in dataset._query_pos_docids.keys():
        avg += len(dataset.get_relevance_docids_by_query(qid))
    return avg/len(dataset._query_pos_docids.keys())

def get_qrel_avg_num_of_rel(query_dic):
    avg = 0
    num_query = 0
    for qid in query_dic.keys():
        num_query += 1
        num_rel = 0
        for docid in query_dic[qid].keys():
            if query_dic[qid][docid] == 1:
                num_rel += 1
        avg += num_rel
    return avg/num_query




dic0 = read_intent_qrel("0.txt")
print(get_qrel_avg_num_of_rel(dic0))
dic1 = read_intent_qrel("1.txt")
print(get_qrel_avg_num_of_rel(dic1))
dic2 = read_intent_qrel("2.txt")
print(get_qrel_avg_num_of_rel(dic2))
dic3 = read_intent_qrel("3.txt")
print(get_qrel_avg_num_of_rel(dic3))

dataset_fold = "../datasets/clueweb09/ClueWeb09-TREC-LTR.txt"
train_set = LetorDataset(dataset_fold, 91, query_level_norm=True, binary_label=True)

# train_set.write_cross_validation_datasets("../datasets/clueweb09", 5)
# print(train_set.get_relevance_docids_by_query("3"))
train_set.update_relevance_label(dic0)
print(get_dataset_avg_num_of_rel(train_set))
train_set.update_relevance_label(dic1)
print(get_dataset_avg_num_of_rel(train_set))
train_set.update_relevance_label(dic2)
print(get_dataset_avg_num_of_rel(train_set))
train_set.update_relevance_label(dic3)
print(get_dataset_avg_num_of_rel(train_set))

# for qid in dic.keys():
#     print(qid, len(dic[qid].keys()), len(train_set.get_candidate_docids_by_query(qid)))

# addtional = {}
# total = 0
# with open("0.txt", 'r') as f:
#     out = open("whitelist.txt", "w")
#     for line in f:
#         qid, _, docid, rel = line.strip().split()
#         _, fold, file, _ = docid.split('-')
#         if fold[:4] != "enwp":
#             if int(fold[2:]) > 11:
#                 out.write(docid + '\n')
#     out.close()


