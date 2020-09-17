import os
import sys
sys.path.append('../')
from dataset.LetorDataset import LetorDataset
from ranker.ProbabilisticRanker import ProbabilisticRanker
from clickModel.SDBN import SDBN
from clickModel.PBM import PBM
from utils import evl_tool
import numpy as np
import multiprocessing as mp
import pickle


def run(train_set, test_set, ranker, num_interation, click_model, num_rankers):
    ndcg_scores = []
    cndcg_scores = []

    query_set = train_set.get_all_querys()
    index = np.random.randint(query_set.shape[0], size=num_interation)
    num_iter = 0
    for i in index:
        qid = query_set[i]
        query_features = train_set.get_all_features_by_query(qid)
        rankers = []
        us = []
        rankers.append(ranker)
        for i in range(num_rankers):
            new_ranker, new_u = ranker.get_new_candidate()
            rankers.append(new_ranker)
            us.append(new_u)

        (inter_list, a) = ranker.probabilistic_multileave(rankers, query_features, 10)

        _, click_label, _ = click_model.simulate(qid, inter_list, train_set)

        outcome = ranker.probabilistic_multileave_outcome(inter_list, rankers, click_label, query_features)
        winners = np.where(np.array(outcome) > outcome[0])

        if np.shape(winners)[1] != 0:
            u = np.zeros(ranker.feature_size)
            for winner in winners[0]:
                u += us[winner - 1]
            u = u / np.shape(winners)[1]
            ranker.update_weights(u, alpha=ranker.learning_rate)

        if num_iter % 1000 == 0:
            all_result = ranker.get_all_query_result_list(test_set, ranker.get_current_weights())
            ndcg = evl_tool.average_ndcg_at_k(test_set, all_result, 10)
            ndcg_scores.append(ndcg)

        cndcg = evl_tool.query_ndcg_at_k(train_set, inter_list, qid, 10)
        cndcg_scores.append(cndcg)
        num_iter += 1

    return ndcg_scores, cndcg_scores


def job(model_type, f, train_set, test_set, delta, alpha, FEATURE_SIZE, num_rankers, output_fold):
    if model_type == "perfect":
        pc = [0.0, 0.2, 0.4, 0.8, 1.0]
        ps = [0.0, 0.0, 0.0, 0.0, 0.0]
    elif model_type == "navigational":
        pc = [0.05, 0.3, 0.5, 0.7, 0.95]
        ps = [0.2, 0.3, 0.5, 0.7, 0.9]
    elif model_type == "informational":
        pc = [0.4, 0.6, 0.7, 0.8, 0.9]
        ps = [0.1, 0.2, 0.3, 0.4, 0.5]
    #
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


    for r in range(1, 26):
        # np.random.seed(r)
        ranker = ProbabilisticRanker(delta, alpha, FEATURE_SIZE)
        print("DBGD fold{} run{} start!".format(f, model_type, r))
        ndcg_scores, cndcg_scores = run(train_set, test_set, ranker, NUM_INTERACTION, cm, num_rankers)
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
        #

if __name__ == "__main__":

    FEATURE_SIZE = 220
    NUM_INTERACTION = 100000
    click_models = ["informational", "perfect"]
    # click_models = ["perfect"]
    # dataset_fold = "../datasets/MSLR10K"
    # dataset_fold = "../datasets/2007_mq_dataset"
    output_fold = "results/istella/DBGD"
    # output_fold = "results/mslr10k/DBGD"
    # output_fold = "results/yahoo/PMGD"
    # taus = [0.1, 0.5, 1.0, 5.0, 10.0]
    alpha = 0.01
    delta = 1
    num_rankers = 1

    # for 5 folds
    for f in range(1, 2):
        # training_path = "{}/Fold{}/train.txt".format(dataset_fold, f)
        # test_path = "{}/Fold{}/test.txt".format(dataset_fold, f)
        training_path = "../datasets/istella/train.txt"
        test_path = "../datasets/istella/test.txt"
        print("loading dataset.....")
        # training_path = "../datasets/ltrc_yahoo/set1.train.txt"
        # test_path = "../datasets/ltrc_yahoo/set1.test.txt"
        train_set = LetorDataset(training_path, FEATURE_SIZE, query_level_norm=True)
        test_set = LetorDataset(test_path, FEATURE_SIZE, query_level_norm=True)

        # for 3 click_models
        for click_model in click_models:
            p = mp.Process(target=job, args=(click_model, f, train_set, test_set,
                                             delta, alpha, FEATURE_SIZE, num_rankers, output_fold))
            p.start()
