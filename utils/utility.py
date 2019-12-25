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