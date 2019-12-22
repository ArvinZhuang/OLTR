import numpy as np

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


def get_scores(weights, features):
    weights = np.array([weights])
    score = np.dot(features, weights.T)[:, 0]
    return score