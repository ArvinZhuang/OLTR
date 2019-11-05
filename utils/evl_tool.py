import numpy as np


def query_ndcg_at_k(dataset, result_list, query, k):
    try:
        pos_docid_set = set(dataset.get_relevance_docids_by_query(query))
    except:
        return 0.0
    dcg = 0.0
    for i in range(0, min(k, len(result_list))):
        docid = result_list[i]
        relevance = dataset.get_relevance_label_by_query_and_docid(query, docid)
        dcg += ((2 ** relevance - 1) / np.log2(i + 2))
    rel_set = []

    for docid in pos_docid_set:
        rel_set.append(dataset.get_relevance_label_by_query_and_docid(query, docid))
    rel_set = sorted(rel_set, reverse=True)
    n = len(pos_docid_set) if len(pos_docid_set) < k else k
    idcg = 0
    for i in range(n):
        idcg += ((2 ** rel_set[i] - 1) / np.log2(i + 2))

    ndcg = (dcg / idcg)
    return ndcg

def average_ndcg_at_k(dataset, query_result_list, k):
    ndcg = 0.0
    num_query = 0
    for query in dataset.get_all_querys():
        try:
            pos_docid_set = set(dataset.get_relevance_docids_by_query(query))
        except:
            # print("Query:", query, "has no relevant document!")
            continue
        dcg = 0.0
        for i in range(0, min(k, len(query_result_list[query]))):
            docid = query_result_list[query][i]
            relevance = dataset.get_relevance_label_by_query_and_docid(query, docid)
            dcg += ((2 ** relevance - 1) / np.log2(i + 2))

        rel_set = []
        for docid in pos_docid_set:
            rel_set.append(dataset.get_relevance_label_by_query_and_docid(query, docid))
        rel_set = sorted(rel_set, reverse=True)
        n = len(pos_docid_set) if len(pos_docid_set) < k else k

        idcg = 0
        for i in range(n):
            idcg += ((2 ** rel_set[i] - 1) / np.log2(i + 2))

        ndcg += (dcg / idcg)
        num_query += 1
    return ndcg / float(num_query)

def get_all_query_ndcg(dataset, query_result_list, k):
    query_ndcg = {}
    for query in dataset.get_all_querys():
        try:
            pos_docid_set = set(dataset.get_relevance_docids_by_query(query))
        except:
            # print("Query:", query, "has no relevant document!")
            query_ndcg[query] = 0
            continue
        dcg = 0.0
        for i in range(0, min(k, len(query_result_list[query]))):
            docid = query_result_list[query][i]
            relevance = dataset.get_relevance_label_by_query_and_docid(query, docid)
            dcg += ((2 ** relevance - 1) / np.log2(i + 2))

        rel_set = []
        for docid in pos_docid_set:
            rel_set.append(dataset.get_relevance_label_by_query_and_docid(query, docid))
        rel_set = sorted(rel_set, reverse=True)
        n = len(pos_docid_set) if len(pos_docid_set) < k else k

        idcg = 0
        for i in range(n):
            idcg += ((2 ** rel_set[i] - 1) / np.log2(i + 2))

        ndcg = (dcg / idcg)
        query_ndcg[query] = ndcg
    return query_ndcg
