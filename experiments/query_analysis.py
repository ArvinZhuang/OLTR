import sys
sys.path.append('../')
from dataset.LetorDataset import LetorDataset
from ranker.PDGDLinearRanker import PDGDLinearRanker, LinearRanker
from clickModel.SDBN import SDBN
from utils import evl_tool
import numpy as np
import multiprocessing as mp
import pickle
import matplotlib.pyplot as plt
#%%
path = "./results/exploration/mq2007/PDGD"

query_ndcg = None
for r in range(1, 26):
    with open("{}/fold{}/{}_tau{}_run{}_final_weight.txt".format(path, 1, 'informational',
                                                                 1.0, r),"rb") as fp:
        weights = pickle.load(fp)

    dataset_fold = "./datasets/2007_mq_dataset"
    training_path = "{}/Fold{}/train.txt".format(dataset_fold, 1)
    train_set = LetorDataset(training_path, 46)

    ranker = LinearRanker(46, 0.1, 1.0)
    ranker.assign_weights(weights)

    results = ranker.get_all_query_result_list(train_set)

    temp = evl_tool.get_all_query_ndcg(train_set, results, 10)
    if query_ndcg is None:
        query_ndcg = temp
    else:
        for query in temp.keys():
            query_ndcg[query] = query_ndcg[query] + temp[query]

for query in query_ndcg.keys():
    query_ndcg[query] = query_ndcg[query]/25


#%%
queries = []
ndcgs = []
for key, value in sorted(query_ndcg.items(), key=lambda kv: kv[1], reverse=True):
    queries.append(key)
    ndcgs.append(value)

# %%
plt.bar(queries, ndcgs)
plt.xlabel('queries', fontsize=5)
plt.ylabel('ndcgs', fontsize=5)
plt.xticks(queries, queries, fontsize=5, rotation=30)
plt.title('doc sample probs')
plt.show()
#%%
low_perform_queries = set(queries[700:])
#%%

def run(train_set, test_set, ranker1, ranker2, num_interation, click_model):
    query_set = train_set.get_all_querys()
    index = np.random.randint(query_set.shape[0], size=num_interation)

    for i in index:
        qid = query_set[i]
        if qid in low_perform_queries:
            ranker1.set_learning_rate(0.01)
            ranker2.set_learning_rate(0.1)

            result_list, scores2 = ranker2.get_query_result_list(train_set, qid)
            _, scores1 = ranker1.get_query_result_list(train_set, qid)

            clicked_doc, click_label = click_model.simulate(qid, result_list)

            ranker2.update_to_clicks(click_label, result_list, scores2, train_set.get_all_features_by_query(qid))
            ranker1.update_to_clicks(click_label, result_list, scores1, train_set.get_all_features_by_query(qid))

        else:
            ranker1.set_learning_rate(0.1)
            ranker2.set_learning_rate(0.01)

            result_list, scores1 = ranker1.get_query_result_list(train_set, qid)
            _, scores2 = ranker2.get_query_result_list(train_set, qid)

            clicked_doc, click_label = click_model.simulate(qid, result_list)

            ranker2.update_to_clicks(click_label, result_list, scores2, train_set.get_all_features_by_query(qid))
            ranker1.update_to_clicks(click_label, result_list, scores1, train_set.get_all_features_by_query(qid))

        # if qid in low_perform_queries:
        #
        #     result_list, scores2 = ranker2.get_query_result_list(train_set, qid)
        #
        #     clicked_doc, click_label = click_model.simulate(qid, result_list)
        #
        #     ranker2.update_to_clicks(click_label, result_list, scores2, train_set.get_all_features_by_query(qid))
        # else:
        #
        #     result_list, scores1 = ranker1.get_query_result_list(train_set, qid)
        #
        #     clicked_doc, click_label = click_model.simulate(qid, result_list)
        #
        #     ranker1.update_to_clicks(click_label, result_list, scores1, train_set.get_all_features_by_query(qid))

    return ranker1.get_current_weights(), ranker2.get_current_weights()


def job(model_type, f, train_set, test_set, tau, r):
    if model_type == "perfect":
        pc = [0.0, 0.5, 1.0]
        ps = [0.0, 0.0, 0.0]
    elif model_type == "navigational":
        pc = [0.05, 0.5, 0.95]
        ps = [0.2, 0.5, 0.9]
    elif model_type == "informational":
        pc = [0.4, 0.7, 0.9]
        ps = [0.1, 0.3, 0.5]

    cm = SDBN(train_set, pc, ps)
        # np.random.seed(r)
    ranker1 = PDGDLinearRanker(FEATURE_SIZE, Learning_rate, tau)
    ranker2 = PDGDLinearRanker(FEATURE_SIZE, Learning_rate, tau)
    print("PDGD tau{} fold{} {} run{} start!".format(tau, f, model_type, r))
    final_weight1,  final_weight2= run(train_set, test_set, ranker1, ranker2, NUM_INTERACTION, cm)
    with open(
            "./results/multiple_ranker/mq2007/PDGD/fold{}/{}_tau{}_run{}_ranker1_weights.txt".format(f, model_type, tau, r),
            "wb") as fp:
        pickle.dump(final_weight1, fp)
    with open(
            "./results/multiple_ranker/mq2007/PDGD/fold{}/{}_tau{}_run{}_ranker2_weights.txt".format(f, model_type, tau, r),
            "wb") as fp:
        pickle.dump(final_weight2, fp)
    print("PDGD tau{} fold{} {} run{} finished!".format(tau, f, model_type, r))

#%%
if __name__ == "__main__":

    FEATURE_SIZE = 46
    NUM_INTERACTION = 10000
    # click_models = ["informational", "navigational", "perfect"]
    click_models = ["informational"]
    Learning_rate = 0.1
    dataset_fold = "./datasets/2007_mq_dataset"
    output_fold = "mq2007"
    taus = [1.0]
    # for 5 folds
    for f in range(1, 2):
        training_path = "{}/Fold{}/train.txt".format(dataset_fold, f)
        test_path = "{}/Fold{}/test.txt".format(dataset_fold, f)
        train_set = LetorDataset(training_path, FEATURE_SIZE)
        test_set = LetorDataset(test_path, FEATURE_SIZE)

        # for 3 click_models
        for click_model in click_models:
            for tau in taus:
                for r in range(1, 26):
                    mp.Process(target=job, args=(click_model, f, train_set, test_set, tau, r)).start()




#%%
path = "./results/multiple_ranker/mq2007/PDGD"

query_ndcg1 = None
query_ndcg2 = None
for r in range(1, 26):
    with open("{}/fold{}/{}_tau{}_run{}_ranker1_weights.txt".format(path, 1, 'informational',
                                                                 1.0, r),"rb") as fp:
        weights1 = pickle.load(fp)
    with open("{}/fold{}/{}_tau{}_run{}_ranker2_weights.txt".format(path, 1, 'informational',
                                                                 1.0, r),"rb") as fp:
        weights2 = pickle.load(fp)
    dataset_fold = "./datasets/2007_mq_dataset"
    training_path = "{}/Fold{}/train.txt".format(dataset_fold, 1)
    train_set = LetorDataset(training_path, 46)

    ranker1 = LinearRanker(46, 0.1, 1.0)
    ranker2 = LinearRanker(46, 0.1, 1.0)
    ranker1.assign_weights(weights1)
    ranker2.assign_weights(weights2)

    results1 = ranker1.get_all_query_result_list(train_set)
    results2 = ranker2.get_all_query_result_list(train_set)

    temp = evl_tool.get_all_query_ndcg(train_set, results1, 10)
    if query_ndcg1 is None:
        query_ndcg1 = temp
    else:
        for query in temp.keys():
            query_ndcg1[query] = query_ndcg1[query] + temp[query]

    temp = evl_tool.get_all_query_ndcg(train_set, results2, 10)
    if query_ndcg2 is None:
        query_ndcg2 = temp
    else:
        for query in temp.keys():
            query_ndcg2[query] = query_ndcg2[query] + temp[query]


#%%
two_ranker_ndcg = []
for query in queries:
    if query in low_perform_queries:
        two_ranker_ndcg.append(query_ndcg2[query]/25)
    else:
        two_ranker_ndcg.append(query_ndcg1[query]/25)
#%%

plt.bar(queries, two_ranker_ndcg)
plt.xlabel('queries', fontsize=5)
plt.ylabel('ndcgs', fontsize=5)
plt.xticks(queries, queries, fontsize=5, rotation=30)
plt.title('doc sample probs')
plt.show()
#%%
print(len(queries))
print(sum(two_ranker_ndcg)/len(two_ranker_ndcg))
print(sum(ndcgs)/len(ndcgs))

#%%

ranker1_ndcg = []
ranker2_ndcg = []
for query in queries:
    ranker1_ndcg.append(query_ndcg1[query]/25)
    ranker2_ndcg.append(query_ndcg2[query] / 25)
#%%
plt.bar(queries, ranker2_ndcg)
plt.xlabel('queries', fontsize=5)
plt.ylabel('ndcgs', fontsize=5)
plt.xticks(queries, queries, fontsize=5, rotation=30)
plt.title('doc sample probs')
plt.show()
#%%
print(len(queries))
print(sum(ranker1_ndcg)/len(ranker1_ndcg))
print(sum(ranker2_ndcg)/len(ranker2_ndcg))
print(sum(ndcgs)/len(ndcgs))

#%%
path = "./results/exploration/mq2007/PDGD"
test_query_ndcg = None
for r in range(1, 26):
    with open("{}/fold{}/{}_tau{}_run{}_final_weight.txt".format(path, 1, 'informational',
                                                                 1.0, r),"rb") as fp:
        weights = pickle.load(fp)

    dataset_fold = "./datasets/2007_mq_dataset"
    test_path = "{}/Fold{}/test.txt".format(dataset_fold, 1)
    test_set = LetorDataset(test_path, 46)

    ranker = LinearRanker(46, 0.1, 1.0)
    ranker.assign_weights(weights)

    results = ranker.get_all_query_result_list(test_set)

    temp = evl_tool.get_all_query_ndcg(test_set, results, 10)
    if test_query_ndcg is None:
        test_query_ndcg = temp
    else:
        for query in temp.keys():
            test_query_ndcg[query] = test_query_ndcg[query] + temp[query]

for query in test_query_ndcg.keys():
    test_query_ndcg[query] = test_query_ndcg[query]/25
#%%
test_queries = []
test_ndcgs = []
for key, value in sorted(test_query_ndcg.items(), key=lambda kv: kv[1], reverse=True):
    test_queries.append(key)
    test_ndcgs.append(value)

# %%
plt.bar(test_queries, test_ndcgs)
plt.xlabel('queries', fontsize=5)
plt.ylabel('ndcgs', fontsize=5)
plt.xticks(test_queries, test_queries, fontsize=5, rotation=30)
plt.title('doc sample probs')
plt.show()
#%%
test_low_perform_queries = set(test_queries[200:])
#%%
path = "./results/multiple_ranker/mq2007/PDGD"

test_query_ndcg1 = None
test_query_ndcg2 = None
for r in range(1, 26):
    with open("{}/fold{}/{}_tau{}_run{}_ranker1_weights.txt".format(path, 1, 'informational',
                                                                 1.0, r),"rb") as fp:
        weights1 = pickle.load(fp)
    with open("{}/fold{}/{}_tau{}_run{}_ranker2_weights.txt".format(path, 1, 'informational',
                                                                 1.0, r),"rb") as fp:
        weights2 = pickle.load(fp)
    dataset_fold = "./datasets/2007_mq_dataset"
    test_path = "{}/Fold{}/test.txt".format(dataset_fold, 1)
    test_set = LetorDataset(test_path, 46)

    ranker1 = LinearRanker(46, 0.1, 1.0)
    ranker2 = LinearRanker(46, 0.1, 1.0)
    ranker1.assign_weights(weights1)
    ranker2.assign_weights(weights2)

    results1 = ranker1.get_all_query_result_list(test_set)
    results2 = ranker2.get_all_query_result_list(test_set)

    temp = evl_tool.get_all_query_ndcg(test_set, results1, 10)
    if test_query_ndcg1 is None:
        test_query_ndcg1 = temp
    else:
        for query in temp.keys():
            test_query_ndcg1[query] = test_query_ndcg1[query] + temp[query]

    temp = evl_tool.get_all_query_ndcg(test_set, results2, 10)
    if test_query_ndcg2 is None:
        test_query_ndcg2 = temp
    else:
        for query in temp.keys():
            test_query_ndcg2[query] = test_query_ndcg2[query] + temp[query]


#%%
test_two_ranker_ndcg = []
for query in test_queries:
    if query in test_low_perform_queries:
        test_two_ranker_ndcg.append(test_query_ndcg2[query]/25)
    else:
        test_two_ranker_ndcg.append(test_query_ndcg1[query]/25)
#%%
plt.bar(test_queries, test_two_ranker_ndcg)
plt.xlabel('queries', fontsize=5)
plt.ylabel('ndcgs', fontsize=5)
plt.xticks(test_queries, test_queries, fontsize=5, rotation=30)
plt.title('doc sample probs')
plt.show()
#%%
print(len(test_queries))
print(sum(test_two_ranker_ndcg)/len(test_two_ranker_ndcg))
print(sum(test_ndcgs)/len(test_ndcgs))
#%%