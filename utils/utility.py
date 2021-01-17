import numpy as np
import requests
import copy
import random

def get_all_query_result_list(weights, dataset):
    query_result_list = {}

    for query in dataset.get_all_querys():
        docid_list = np.array(dataset.get_candidate_docids_by_query(query))
        docid_list = docid_list.reshape((len(docid_list), 1))
        feature_matrix = dataset.get_all_features_by_query(query)
        score_list = get_scores(weights, feature_matrix)

        docid_score_list = np.column_stack((docid_list, score_list))
        docid_score_list = np.flip(docid_score_list[docid_score_list[:, 1].argsort()], 0)

        query_result_list[query] = docid_score_list[:, 0]

    return query_result_list


def get_query_result_list(weights, dataset, query):
    docid_list = dataset.get_candidate_docids_by_query(query)
    feature_matrix = dataset.get_all_features_by_query(query)

    score_list = get_scores(weights, feature_matrix)

    docid_score_list = zip(docid_list, score_list)
    docid_score_list = sorted(docid_score_list, key=lambda x: x[1], reverse=True)

    query_result_list = []
    for i in range(0, len(docid_list)):
        (docid, socre) = docid_score_list[i]
        query_result_list.append(docid)
    return query_result_list


def get_scores(weights, features):
    weights = np.array([weights])
    score = np.dot(features, weights.T)[:, 0]
    return score


def send_progress(name, current, total, comment):
    url = "https://ielab-sysrev1.uqcloud.net?name={}&current={}&total={}&comment={}".format(name, current, total,
                                                                                            comment)
    try:
        requests.put(url)
        success = True
    except:
        success = False
    return success


def get_DCG_rewards(click_labels, propensities, method="both"):
    reward = np.zeros(len(click_labels))

    for iPos in range(len(click_labels)):
        reward[iPos] = 1 / np.log2(iPos + 2.0)

        if click_labels[iPos] == 1:
            # reward[iPos] = 0
            if method == "positive":
                reward[iPos] = reward[iPos] / propensities[iPos]
            if method == "negative":
                reward[iPos] = (reward[iPos] / propensities[iPos]) - reward[iPos]
            if method == "both":
                reward[iPos] = 2 * (reward[iPos] / propensities[iPos]) - reward[iPos]
            # reward[iPos] = reward[iPos] / np.max([propensities[iPos], 0.5])  # propensity clipping
        else:
            # unbiased negative rewards
            if method == "positive":
                reward[iPos] = 0
            if method == "negative":
                reward[iPos] = -reward[iPos]
            if method == "both":
                reward[iPos] = -reward[iPos]
    return reward

# def get_DCG_MDPrewards(click_labels, propensities, method="both", gamma=0.5):
#     M = len(click_labels)
#     discounts = np.logspace(0, M - 1, M, base=gamma)
#     MDP_rewards = np.zeros(M)
#
#
#     for t in range(M):
#         MDP_rewards[t] = np.sum(discounts[:M-t] * get_DCG_rewards(click_labels, propensities, method))
#         click_labels = np.delete(click_labels, 0)
#
#     return MDP_rewards

def get_real_DCGs(query, result_list, dataset):
    result_list = result_list[:10]
    rels = np.array(dataset.get_all_relevance_label_by_query(query))
    labels = rels[result_list]

    DCGs = np.zeros(len(labels))
    for iPos in range(len(labels)):
        if iPos == 0:
            DCGs[iPos] = labels[iPos]
        else:
            DCGs[iPos] = labels[iPos] / np.log(iPos + 1.0)

    return DCGs

def get_DCG_MDPrewards(click_labels, propensities, method="both", gamma=0.99):
    M = len(click_labels)
    MDP_rewards = np.zeros(M)
    DCG_rewards = get_DCG_rewards(click_labels, propensities, method)

    discounts = np.logspace(0, M - 1, M, base=gamma)
    for t in range(M):
        MDP_rewards[t] = np.sum(discounts[:M-t] * DCG_rewards[t:])

    return MDP_rewards


def GetReward_ARP(rates, propensities):
    reward = np.zeros(len(rates))
    for iPos in range(len(rates)):
        if iPos == 0:
            reward[iPos] = (2 ** rates[iPos] - 1)
        else:
            reward[iPos] = (2 ** rates[iPos] - 1) / np.log(iPos + 1.0)
        reward[iPos] = reward[iPos] / np.max([propensities[iPos], 0.5])

    return reward


def read_intent_qrel(path: str):
    # q-d pair dictionary
    qrel_dic = {}

    with open(path, 'r') as f:
        for line in f:
            qid, _, docid, rel = line.strip().split()
            if qid in qrel_dic.keys():
                qrel_dic[qid][docid] = int(rel)
            else:
                qrel_dic[qid] = {docid: int(rel)}
    return qrel_dic


def get_intent_dataset(train_set, test_set, intent_path):
    new_train_set = copy.deepcopy(train_set)
    new_test_set = copy.deepcopy(test_set)
    qrel_dic = read_intent_qrel(intent_path)
    new_train_set.update_relevance_label(qrel_dic)
    new_test_set.update_relevance_label(qrel_dic)
    return new_train_set, new_test_set


def get_groups_dataset(train_set, intent_paths, num_groups=4):
    qrel_dics = []

    print("Reading intents......")
    for path in intent_paths:
        qrel_dics.append(read_intent_qrel(path))

    print("Randomly assign groups......")
    for qid in qrel_dics[0].keys():
        qid_rel_lists = []
        for qrel_dic in qrel_dics:
            doc_rels = {}
            for docid in qrel_dic[qid].keys():
                doc_rels[docid] = qrel_dic[qid][docid]
            qid_rel_lists.append(doc_rels)

        random.shuffle(qid_rel_lists)
        for i in range(len(qrel_dics)):
            for docid in qrel_dics[i][qid].keys():
                qrel_dics[i][qid][docid] = qid_rel_lists[i][docid]

    datasets = []
    print("Generating new datasets......")
    for qrel_dic in qrel_dics:
        new_train_set = copy.deepcopy(train_set)
        new_train_set.update_relevance_label(qrel_dic)
        datasets.append(new_train_set)
    return datasets[:num_groups]