import sys

sys.path.append('../')
from dataset.LetorDataset import LetorDataset
from ranker.ProbabilisticRanker import ProbabilisticRanker
from clickModel.SDBN import SDBN
from utils.utility import get_groups_dataset
from utils import evl_tool
import numpy as np
import multiprocessing as mp
import pickle
import copy
import os
import random


def run(train_intents, ranker, num_interation, click_model, num_rankers):

    ndcg_scores = [[], [], [], [], []]
    cndcg_scores = []


    query_set = train_set.get_all_querys()
    index = np.random.randint(query_set.shape[0], size=num_interation)
    num_iter = 0

    current_train_set = train_intents[0]

    for i in index:
        if num_iter % 50000 == 0 and num_iter > 0:
            # print("Change intent to", int(num_iter/10000))
            all_result = ranker.get_all_query_result_list(current_train_set, ranker.get_current_weights())
            ndcg = evl_tool.average_ndcg_at_k(current_train_set, all_result, 10)
            ndcg_scores[0].append(ndcg)

            current_train_set = train_intents[int(num_iter / 50000)]


        qid = query_set[i]

        query_features = current_train_set.get_all_features_by_query(qid)
        rankers = []
        us = []
        rankers.append(ranker)
        for i in range(num_rankers):
            new_ranker, new_u = ranker.get_new_candidate()
            rankers.append(new_ranker)
            us.append(new_u)

        (inter_list, a) = ranker.probabilistic_multileave(rankers, query_features, 10)

        _, click_label, _ = click_model.simulate(qid, inter_list, current_train_set)

        outcome = ranker.probabilistic_multileave_outcome(inter_list, rankers, click_label, query_features)
        winners = np.where(np.array(outcome) > outcome[0])

        if np.shape(winners)[1] != 0:
            u = np.zeros(ranker.feature_size)
            for winner in winners[0]:
                u += us[winner - 1]
            u = u / np.shape(winners)[1]
            ranker.update_weights(u, alpha=ranker.learning_rate)

        if num_iter % 1000 == 0:
            all_result = ranker.get_all_query_result_list(current_train_set, ranker.get_current_weights())
            ndcg = evl_tool.average_ndcg_at_k(current_train_set, all_result, 10)
            ndcg_scores[0].append(ndcg)

            for intent in range(4):
                all_result = ranker.get_all_query_result_list(train_intents[intent], ranker.get_current_weights())
                ndcg = evl_tool.average_ndcg_at_k(train_intents[intent], all_result, 10)
                ndcg_scores[intent + 1].append(ndcg)

        cndcg = evl_tool.query_ndcg_at_k(current_train_set, inter_list, qid, 10)
        cndcg_scores.append(cndcg)

        num_iter += 1

    return ndcg_scores, cndcg_scores


def job(model_type, f, train_set, intent_paths, delta, alpha, FEATURE_SIZE, num_rankers, output_fold):
    if model_type == "perfect":
        pc = [0.0, 1.0]
        ps = [0.0, 0.0]

    elif model_type == "navigational":
        pc = [0.05, 0.95]
        ps = [0.2, 0.9]

    elif model_type == "informational":
        pc = [0.3, 0.7]
        ps = [0.1, 0.5]
    # cm = PBM(pc, 1)
    cm = SDBN(pc, ps)

    for r in range(1, 26):
        random.seed(r)
        np.random.seed(r)
        datasets = get_groups_dataset(train_set, intent_paths)
        ranker = ProbabilisticRanker(delta, alpha, FEATURE_SIZE)

        print("PMGD intent change {} fold{} run{} start!".format(model_type, f, r))
        ndcg_scores, cndcg_scores = run(datasets, ranker, NUM_INTERACTION, cm, num_rankers)

        # create directory if not exist
        os.makedirs(os.path.dirname("{}/current_intent/fold{}/".format(output_fold, f)), exist_ok=True)
        with open(
                "{}/current_intent/fold{}/{}_run{}_cndcg.txt".format(output_fold, f, model_type, r),
                "wb") as fp:
            pickle.dump(cndcg_scores, fp)

        with open(
                "{}/current_intent/fold{}/{}_run{}_ndcg.txt".format(output_fold, f, model_type, r),
                "wb") as fp:
            pickle.dump(ndcg_scores[0], fp)

        for i in range(len(ndcg_scores) - 1):  # the intent ndcg start from 1.
            os.makedirs(os.path.dirname("{}/intent{}/fold{}/".format(output_fold, i + 1, f)),
                        exist_ok=True)  # create directory if not exist\

            with open(
                    "{}/intent{}/fold{}/{}_run{}_ndcg.txt".format(output_fold, i + 1, f, model_type, r),
                    "wb") as fp:
                pickle.dump(ndcg_scores[i + 1], fp)

        print("PMGD intent change {} run{} finish!".format(model_type, r))
        print()


if __name__ == "__main__":

    FEATURE_SIZE = 105
    NUM_INTERACTION = 200000
    click_models = ["informational", "navigational", "perfect"]
    # click_models = ["informational"]
    alpha = 0.01
    delta = 1
    num_rankers = 1

    dataset_path = "datasets/clueweb09_intent_change.txt"
    intent_path = "intents"
    output_fold = "results/SDBN/PMGD/abrupt_group_change_50k"

    train_set = LetorDataset(dataset_path, FEATURE_SIZE, query_level_norm=True, binary_label=True)

    intent_paths = ["{}/1.txt".format(intent_path),
                    "{}/2.txt".format(intent_path),
                    "{}/3.txt".format(intent_path),
                    "{}/4.txt".format(intent_path)]

    for click_model in click_models:
        p = mp.Process(target=job, args=(click_model, 1, train_set, intent_paths,
                                         delta, alpha, FEATURE_SIZE, num_rankers, output_fold))
        p.start()
