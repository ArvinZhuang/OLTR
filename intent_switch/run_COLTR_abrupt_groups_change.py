import sys
sys.path.append('../')
from dataset.LetorDataset import LetorDataset
from ranker.COLTRLinearRanker import COLTRLinearRanker
from clickModel.SDBN import SDBN
from utils.utility import get_groups_dataset
from utils import evl_tool
import numpy as np
import multiprocessing as mp
import pickle
import os
import random


def run(train_intents, ranker, num_interation, click_model, num_rankers):
    ndcg_scores = [[], [], [], [], []]
    cndcg_scores = []

    query_set = train_intents[0].get_all_querys()
    index = np.random.randint(query_set.shape[0], size=num_interation)
    num_iter = 0

    current_train_set = train_intents[0]

    for i in index:
        if num_iter % 50000 == 0 and num_iter > 0:
            # print("Change intent to", int(num_iter/10000))
            all_result = ranker.get_all_query_result_list(current_train_set)
            ndcg = evl_tool.average_ndcg_at_k(current_train_set, all_result, 10)
            ndcg_scores[0].append(ndcg)
            current_train_set = train_intents[int(num_iter / 50000)]


        qid = query_set[i]
        result_list = ranker.get_query_result_list(current_train_set, qid)

        clicked_doc, click_label, _ = click_model.simulate(qid, result_list, current_train_set)

        # if no clicks, skip.
        if len(clicked_doc) == 0:
            if num_iter % 1000 == 0:
                all_result = ranker.get_all_query_result_list(current_train_set)
                ndcg = evl_tool.average_ndcg_at_k(current_train_set, all_result, 10)
                ndcg_scores[0].append(ndcg)

                for intent in range(4):
                    all_result = ranker.get_all_query_result_list(train_intents[intent])
                    ndcg = evl_tool.average_ndcg_at_k(train_intents[intent], all_result, 10)
                    ndcg_scores[intent + 1].append(ndcg)

            cndcg = evl_tool.query_ndcg_at_k(current_train_set, result_list, qid, 10)
            cndcg_scores.append(cndcg)
            # print(num_inter, ndcg, "continue")
            num_iter += 1
            continue

        # flip click label. exp: [1,0,1,0,0] -> [0,1,0,0,0]
        last_click = np.where(click_label == 1)[0][-1]
        click_label[:last_click + 1] = 1 - click_label[:last_click + 1]

        # bandit record
        record = (qid, result_list, click_label, ranker.get_current_weights())

        unit_vectors = ranker.sample_unit_vectors(num_rankers)
        canditate_rankers = ranker.sample_canditate_rankers(
            unit_vectors)  # canditate_rankers are ranker weights, not ranker class

        # winner_rankers are index of candidates rankers who win the evaluation
        winner_rankers = ranker.infer_winners(canditate_rankers[:num_rankers],
                                              record)

        if winner_rankers is not None:
            gradient = np.sum(unit_vectors[winner_rankers - 1], axis=0) / winner_rankers.shape[0]
            ranker.update(gradient)

        if num_iter % 1000 == 0:
            all_result = ranker.get_all_query_result_list(current_train_set)
            ndcg = evl_tool.average_ndcg_at_k(current_train_set, all_result, 10)
            ndcg_scores[0].append(ndcg)

            for intent in range(4):
                all_result = ranker.get_all_query_result_list(train_intents[intent])
                ndcg = evl_tool.average_ndcg_at_k(train_intents[intent], all_result, 10)
                ndcg_scores[intent + 1].append(ndcg)

        cndcg = evl_tool.query_ndcg_at_k(current_train_set, result_list, qid, 10)
        cndcg_scores.append(cndcg)
        # print(num_inter, ndcg)
        num_iter += 1

    return ndcg_scores, cndcg_scores


def job(model_type, f, train_set, intent_paths, tau, step_size, gamma, num_rankers, learning_rate_decay, output_fold):
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
        ranker = COLTRLinearRanker(FEATURE_SIZE, Learning_rate, step_size, tau, gamma, learning_rate_decay=learning_rate_decay)

        print("COLTR intent change {} fold{} run{} start!".format(model_type, f, r))
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

        print("COLTR intent change {} run{} finish!".format(model_type, r))
        print()


if __name__ == "__main__":

    FEATURE_SIZE = 105
    NUM_INTERACTION = 200000
    click_models = ["informational", "navigational", "perfect"]
    # click_models = ["perfect"]
    Learning_rate = 0.1
    num_rankers = 499
    tau = 0.1
    gamma = 1
    learning_rate_decay = 0.99966
    step_size = 1

    dataset_path = "datasets/clueweb09_intent_change.txt"
    intent_path = "intents"
    output_fold = "results/SDBN/COLTR/abrupt_group_change_lrdecay_50k"

    train_set = LetorDataset(dataset_path, FEATURE_SIZE, query_level_norm=True, binary_label=True)

    intent_paths = ["{}/1.txt".format(intent_path),
                    "{}/2.txt".format(intent_path),
                    "{}/3.txt".format(intent_path),
                    "{}/4.txt".format(intent_path)]

    # for 3 click_models
    processors = []
    for click_model in click_models:
        p = mp.Process(target=job, args=(click_model, 1, train_set, intent_paths,
                                         tau, step_size, gamma, num_rankers, learning_rate_decay, output_fold))
        p.start()
        processors.append(p)
    for p in processors:
        p.join()
