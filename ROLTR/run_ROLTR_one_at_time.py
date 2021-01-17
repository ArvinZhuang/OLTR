import os
import sys

sys.path.append('../')
from dataset import LetorDataset
import numpy as np
from clickModel.PBM import PBM
from ranker.MDPRankerV2 import MDPRankerV2
from utils import evl_tool
from utils.utility import get_DCG_rewards, get_DCG_MDPrewards
import multiprocessing as mp
import pickle

NUM_INTERACTION = 100000

# %%
def run(train_set, test_set, ranker, eta, gamma, reward_method, num_interation, click_model):
    ndcg_scores = []
    cndcg_scores = []
    query_set = train_set.get_all_querys()
    index = np.random.randint(query_set.shape[0], size=num_interation)
    num_iter = 0
    for i in index:
        qid = query_set[i]

        displayed_docids = []
        for rank in range(10):
            result_list = ranker.get_query_result_list(train_set, qid)

            if len(displayed_docids) == len(result_list):
                break

            for docid in result_list:
                if docid not in displayed_docids:
                    displayed_docids.append(docid)
                    break

            candidate_docids = []
            for id in result_list:
                if id == docid or id not in displayed_docids:
                    candidate_docids.append(id)

            click_label, propensity = click_model.simulate_with_position(qid, docid, train_set, rank)

            dcg = 1 / np.log2(rank + 2.0)
            neg_reward = dcg * (click_label - 1) + ((1-propensity)/propensity) * dcg * click_label
            pos_reward = dcg / propensity * click_label
            reward = pos_reward + neg_reward
            #
            ranker.update_policy(qid, candidate_docids, [reward], train_set)

        if num_iter % 1000 == 0:
            all_result = ranker.get_all_query_result_list(test_set)
            ndcg = evl_tool.average_ndcg_at_k(test_set, all_result, 10)
            ndcg_scores.append(ndcg)

            # print(num_iter, ndcg)
        cndcg = evl_tool.query_ndcg_at_k(train_set, displayed_docids, qid, 10)
        cndcg_scores.append(cndcg)

        # all_result = ranker.get_all_query_result_list(test_set)
        # ndcg = evl_tool.average_ndcg_at_k(test_set, all_result, 10)
        # print(num_iter, ndcg)
        num_iter += 1
    return ndcg_scores, cndcg_scores


def job(model_type, learning_rate, eta, gamma, reward_method, f, train_set, test_set, num_features, output_fold):

    if model_type == "perfect":
        pc = [0.0, 0.2, 0.4, 0.8, 1.0]
        ps = [0.0, 0.0, 0.0, 0.0, 0.0]
    elif model_type == "navigational":
        pc = [0.05, 0.3, 0.5, 0.7, 0.95]
        ps = [0.2, 0.3, 0.5, 0.7, 0.9]
    elif model_type == "informational":
        pc = [0.4, 0.6, 0.7, 0.8, 0.9]
        ps = [0.1, 0.2, 0.3, 0.4, 0.5]
    # #
    # if model_type == "perfect":
    #     pc = [0.0, 0.5, 1.0]
    #     ps = [0.0, 0.0, 0.0]
    # elif model_type == "navigational":
    #     pc = [0.05, 0.5, 0.95]
    #     ps = [0.2, 0.5, 0.9]
    # elif model_type == "informational":
    #     pc = [0.4, 0.7, 0.9]
    #     ps = [0.1, 0.3, 0.5]
    cm = PBM(pc, 1)

    for r in range(1, 16):
        # np.random.seed(r)
        ranker = MDPRankerV2(256, num_features, learning_rate, Lenepisode=1)
        print("MDP Adam MSLR10K fold{} {} eta{} reward {} run{} start!".format(f, model_type, eta, reward_method, r))
        ndcg_scores, cndcg_scores = run(train_set, test_set, ranker, eta, gamma, reward_method, NUM_INTERACTION, cm)
        os.makedirs(os.path.dirname("{}/fold{}/".format(output_fold, f)),
                    exist_ok=True)  # create directory if not exist
        with open(
                "{}/fold{}/{}_run{}_ndcg.txt".format(output_fold, f, model_type, r),
                "wb") as fp:
            pickle.dump(ndcg_scores, fp)
        with open(
                "{}/fold{}/{}_run{}_cndcg.txt".format(output_fold, f, model_type, r),
                "wb") as fp:
            pickle.dump(cndcg_scores, fp)

        print("MDP MSLR10K fold{} {} eta{} reward{} run{} done!".format(f, model_type, eta, reward_method, r))


if __name__ == "__main__":

    FEATURE_SIZE = 136
    NUM_INTERACTION = 100000
    learning_rate = 0.01
    eta = 1
    gamma = 0.0
    reward_method = "both"
    click_models = ["informational", "perfect"]
    # click_models = ["informational"]
    # dataset_fold = "../datasets/2007_mq_dataset"
    dataset_fold = "../datasets/MSLR10K"
    output_fold = "results/mslr10k/MDP_with_SGD_optimizer/MDP_001_both_one_at_time"
    print("reward:", reward_method, "lr:", learning_rate, "eta:", eta, output_fold, "gamma", gamma)
    # for 5 folds
    for f in range(1, 6):
        # training_path = "{}/set1.train.txt".format(dataset_fold)
        # test_path = "{}/set1.test.txt".format(dataset_fold)
        training_path = "{}/Fold{}/train.txt".format(dataset_fold, f)
        test_path = "{}/Fold{}/test.txt".format(dataset_fold, f)
        train_set = LetorDataset(training_path, FEATURE_SIZE, query_level_norm=True, cache_root="../datasets/cache")
        test_set = LetorDataset(test_path, FEATURE_SIZE, query_level_norm=True, cache_root="../datasets/cache")

        for click_model in click_models:
            p = mp.Process(target=job, args=(click_model, learning_rate, eta, gamma, reward_method, f, train_set, test_set, FEATURE_SIZE, output_fold))
            p.start()

