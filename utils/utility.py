import numpy as np
import requests

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
    url = "https://ielab-sysrev1.uqcloud.net?name={}&current={}&total={}&comment={}".format(name, current, total, comment)
    try:
        requests.put(url)
        success = True
    except:
        success = False
    return success

def GetReward_DCG(rates, propensities):
    reward = np.zeros(len(rates))
    for iPos in range(len(rates)):
        if iPos == 0:
            reward[iPos] = (2**rates[iPos]-1)
        else:
            reward[iPos] = (2**rates[iPos]-1) / np.log(iPos + 1.0)
        reward[iPos] = reward[iPos] / np.max([propensities[iPos], 0.5])
    propensities = -propensities

    ind = [np.where(reward==0)[0]]
    reward[ind] = propensities[ind]

    return reward

def GetReward_ARP(rates, propensities):
    reward = np.zeros(len(rates))
    for iPos in range(len(rates)):
        if iPos == 0:
            reward[iPos] = (2**rates[iPos]-1)
        else:
            reward[iPos] = (2**rates[iPos]-1) / np.log(iPos + 1.0)
        reward[iPos] = reward[iPos] / np.max([propensities[iPos], 0.5])

    return reward

def GetReturn_DCG(rates, propensities):
    ndoc = len(rates)
    returns = GetReward_DCG(rates, propensities)

    # for iPos in range(len(rates)-1):
    #     returns[ndoc -2 - iPos] += returns[ndoc -1 - iPos]
    return returns