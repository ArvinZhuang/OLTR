import os
import sys

sys.path.append('../')
from dataset import LetorDataset
import numpy as np
from clickModel.PBM import PBM
from ranker.MDPRankerV2 import MDPRankerV2
from utils import evl_tool
from utils.utility import get_DCG_rewards, get_DCG_MDPrewards, get_real_DCGs
import multiprocessing as mp
import pickle


# %%
def run(train_set, test_set, ranker, num_interation):
    ndcg_scores = []
    cndcg_scores = []
    query_set = train_set.get_all_querys()
    index = np.random.randint(query_set.shape[0], size=num_interation)
    num_iter = 0
    for i in index:
        qid = query_set[i]
        result_list = ranker.get_query_result_list(train_set, qid)
        DCGs = get_real_DCGs(qid, result_list, train_set)

        # ranker.record_episode(qid, result_list, rewards)

        ranker.update_policy(qid, result_list, DCGs, train_set)

        if num_iter % 1000 == 0:
            all_result = ranker.get_all_query_result_list(test_set)
            ndcg = evl_tool.average_ndcg_at_k(test_set, all_result, 10)
            ndcg_scores.append(ndcg)
            print(num_iter, ndcg)
        cndcg = evl_tool.query_ndcg_at_k(train_set, result_list, qid, 10)
        cndcg_scores.append(cndcg)

        # all_result = ranker.get_all_query_result_list(test_set)
        # ndcg = evl_tool.average_ndcg_at_k(test_set, all_result, 10)

        num_iter += 1
    return ndcg_scores, cndcg_scores


def job(learning_rate, f, train_set, test_set, num_features, output_fold):


    for r in range(1, 2):
        # np.random.seed(r)
        ranker = MDPRankerV2(256, num_features, learning_rate)
        print("MDP fold{} run{} start!".format(f, r))
        ndcg_scores, cndcg_scores = run(train_set, test_set, ranker, NUM_INTERACTION)
        os.makedirs(os.path.dirname("{}/fold{}/".format(output_fold, f)),
                    exist_ok=True)  # create directory if not exist
        with open(
                "{}/fold{}/{}_run{}_ndcg.txt".format(output_fold, f, "MDPRank", r),
                "wb") as fp:
            pickle.dump(ndcg_scores, fp)
        with open(
                "{}/fold{}/{}_run{}_cndcg.txt".format(output_fold, f, "MDPRank", r),
                "wb") as fp:
            pickle.dump(cndcg_scores, fp)

        print("MDP fold{} run{} done!".format(f, r))


if __name__ == "__main__":

    FEATURE_SIZE = 220
    NUM_INTERACTION = 100000
    learning_rate = 0.01


    # dataset_fold = "../datasets/MSLR10K"
    # output_fold = "results/mslr10k/MDPRank/MDPRank_001_gamma0"
    # dataset_fold = "../datasets/ltrc_yahoo"
    # output_fold = "results/yahoo/MDPRank/MDPRank_0005_gamma0"
    dataset_fold = "../datasets/istella"
    output_fold = "results/istella/MDPRank/MDPRank_001_gamma0"

    # for 5 folds
    for f in range(1, 2):
        training_path = "{}/train.txt".format(dataset_fold)
        test_path = "{}/test.txt".format(dataset_fold)
        # training_path = "{}/set1.train.txt".format(dataset_fold)
        # test_path = "{}/set1.test.txt".format(dataset_fold)
        # training_path = "{}/Fold{}/train.txt".format(dataset_fold, f)
        # test_path = "{}/Fold{}/test.txt".format(dataset_fold, f)
        train_set = LetorDataset(training_path, FEATURE_SIZE, query_level_norm=True)
        test_set = LetorDataset(test_path, FEATURE_SIZE, query_level_norm=True)
        # %%
        processors = []

        p = mp.Process(target=job, args=(learning_rate, f, train_set, test_set, FEATURE_SIZE, output_fold))
        p.start()
        processors.append(p)
    for p in processors:
        p.join()
