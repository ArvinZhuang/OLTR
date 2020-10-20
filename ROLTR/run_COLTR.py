import os
import sys
sys.path.append('../')
from dataset.LetorDataset import LetorDataset
from ranker.COLTRLinearRanker import COLTRLinearRanker
from clickModel.SDBN import SDBN
from clickModel.PBM import PBM
from utils import evl_tool

import numpy as np
import multiprocessing as mp
import pickle
from utils.utility import get_DCG_rewards, get_DCG_MDPrewards


def run(train_set, test_set, ranker, num_interation, click_model, num_rankers):

    ndcg_scores = []
    cndcg_scores = []
    query_set = train_set.get_all_querys()
    index = np.random.randint(query_set.shape[0], size=num_interation)

    num_interation = 0

    # correct = 0
    # wrong = 0
    for i in index:

        qid = query_set[i]

        result_list = ranker.get_query_result_list(train_set, qid)

        clicked_doc, click_label, _ = click_model.simulate(qid, result_list, train_set)

        # if no clicks, skip.
        if len(clicked_doc) == 0:
            if num_interation % 1000 == 0:
                all_result = ranker.get_all_query_result_list(test_set)
                ndcg = evl_tool.average_ndcg_at_k(test_set, all_result, 10)
                ndcg_scores.append(ndcg)

            cndcg = evl_tool.query_ndcg_at_k(train_set, result_list, qid, 10)
            cndcg_scores.append(cndcg)
            num_interation += 1
            continue

        # flip click label. exp: [1,0,1,0,0] -> [0,1,0,0,0]
        last_click = np.where(click_label == 1)[0][-1]
        click_label[:last_click + 1] = 1 - click_label[:last_click + 1]

        # propensities = np.power(np.divide(1, np.arange(1.0, len(click_label) + 1)), 1)
        #
        # # directly using pointwise rewards
        # rewards = get_DCG_rewards(click_label, propensities, "both")

        # bandit record
        record = (qid, result_list, click_label, ranker.get_current_weights())

        unit_vectors = ranker.sample_unit_vectors(num_rankers)
        canditate_rankers = ranker.sample_canditate_rankers(unit_vectors)  # canditate_rankers are ranker weights, not ranker class

        winner_rankers = ranker.infer_winners(canditate_rankers[:num_rankers], record)  # winner_rankers are index of candidates rankers who win the evaluation

        #### This part of code is used to test correctness of counterfactual evaluation ####
        # if winner_rankers is not None:
        #     all_result = utility.get_query_result_list(ranker.get_current_weights(), train_set, qid)
        #     current_ndcg = evl_tool.query_ndcg_at_k(train_set, all_result, qid, 10)
        #     for weights in canditate_rankers[winner_rankers - 1]:
        #         canditate_all_result = utility.get_query_result_list(weights, train_set, qid)
        #         canditate_all_result_ndcg = evl_tool.query_ndcg_at_k(train_set, canditate_all_result, qid, 10)
        #
        #         if canditate_all_result_ndcg >= current_ndcg:
        #             correct += 1
        #         else:
        #             wrong += 1
        #     print(correct, wrong, correct / (correct + wrong))
        ######################################################################################

        if winner_rankers is not None:
            gradient = np.sum(unit_vectors[winner_rankers - 1], axis=0) / winner_rankers.shape[0]
            ranker.update(gradient)

        if num_interation % 1000 == 0:
            all_result = ranker.get_all_query_result_list(test_set)
            ndcg = evl_tool.average_ndcg_at_k(test_set, all_result, 10)
            ndcg_scores.append(ndcg)

        cndcg = evl_tool.query_ndcg_at_k(train_set, result_list, qid, 10)
        cndcg_scores.append(cndcg)
        final_weight = ranker.get_current_weights()
        num_interation += 1

        # print(num_interation, ndcg)

    return ndcg_scores, cndcg_scores


def job(model_type, f, train_set, test_set, tau, step_size, gamma, num_rankers, learning_rate_decay, output_fold):
    # if model_type == "perfect":
    #     pc = [0.0, 0.5, 1.0]
    #     ps = [0.0, 0.0, 0.0]
    # elif model_type == "navigational":
    #     pc = [0.05, 0.5, 0.95]
    #     ps = [0.2, 0.5, 0.9]
    # elif model_type == "informational":
    #     pc = [0.4, 0.7, 0.9]
    #     ps = [0.1, 0.3, 0.5]

    # click setting for 4-grade relevance label datasets.
    if model_type == "perfect":
        pc = [0.0, 0.2, 0.4, 0.8, 1.0]
        ps = [0.0, 0.0, 0.0, 0.0, 0.0]
    elif model_type == "navigational":
        pc = [0.05, 0.3, 0.5, 0.7, 0.95]
        ps = [0.2, 0.3, 0.5, 0.7, 0.9]
    elif model_type == "informational":
        pc = [0.4, 0.6, 0.7, 0.8, 0.9]
        ps = [0.1, 0.2, 0.3, 0.4, 0.5]

    # using PBM click model to simulate clicks.
    # cm = SDBN(pc, ps)
    cm = PBM(pc, 1)
    for r in range(16, 26):
        # np.random.seed(r)
        ranker = COLTRLinearRanker(FEATURE_SIZE, Learning_rate, step_size, tau, gamma, learning_rate_decay=learning_rate_decay)
        print("COTLR {} tau{} fold{} {} run{} start!".format(output_fold, tau, f, model_type, r))

        ndcg_scores, cndcg_scores = run(train_set, test_set, ranker, NUM_INTERACTION, cm, num_rankers)
        os.makedirs(os.path.dirname("{}/fold{}/".format(output_fold, f)),
                    exist_ok=True)  # create directory if not exist
        # store the results.
        with open(
                "{}/fold{}/{}_run{}_ndcg.txt".format(output_fold, f, model_type, r),
                "wb") as fp:
            pickle.dump(ndcg_scores, fp)
        with open(
                "{}/fold{}/{}_run{}_cndcg.txt".format(output_fold, f, model_type, r),
                "wb") as fp:
            pickle.dump(cndcg_scores, fp)


if __name__ == "__main__":

    FEATURE_SIZE = 700
    NUM_INTERACTION = 100000
    click_models = ["informational", "perfect"]
    # click_models = ["perfect"]
    # dataset_fold = "../datasets/MSLR10K"
    # dataset_fold = "../datasets/istella"
    # output_fold = "results/yahoo/COLTR_gamma1"
    output_fold = "results/yahoo/COLTR"

    Learning_rate = 0.1
    num_rankers = 499
    tau = 0.1
    gamma = 1
    learning_rate_decay = 0.99966
    step_size = 1

    for f in range(1, 2):
        # training_path = "{}/train.txt".format(dataset_fold)
        # test_path = "{}/test.txt".format(dataset_fold)
        # training_path = "{}/Fold{}/train.txt".format(dataset_fold, f)
        # test_path = "{}/Fold{}/test.txt".format(dataset_fold, f)
        training_path = "../datasets/ltrc_yahoo/set1.train.txt"
        test_path = "../datasets/ltrc_yahoo/set1.test.txt"
        train_set = LetorDataset(training_path, FEATURE_SIZE, query_level_norm=False)
        test_set = LetorDataset(test_path, FEATURE_SIZE, query_level_norm=False)

        processors = []
        # for click_models
        for click_model in click_models:
            p = mp.Process(target=job, args=(click_model, f, train_set, test_set,
                                             tau, step_size, gamma, num_rankers, learning_rate_decay, output_fold))
            p.start()
            processors.append(p)
    for p in processors:
        p.join()




