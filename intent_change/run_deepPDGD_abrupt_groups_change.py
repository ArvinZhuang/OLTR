import sys

sys.path.append('../')
from dataset.LetorDataset import LetorDataset
from ranker.PDGDNeuralRanker import PDGDNeuralRanker
from clickModel.SDBN import SDBN
from clickModel.PBM import PBM
from utils import evl_tool
from utils.utility import get_groups_dataset
import numpy as np
import multiprocessing as mp
import pickle
import random
import os


def run(train_intents, ranker, num_interation, click_model, group_sequence):
    ndcg_scores = []
    for x in range(len(train_intents) + 1):
        ndcg_scores.append([])
    cndcg_scores = []

    query_set = train_intents[0].get_all_querys()
    index = np.random.randint(query_set.shape[0], size=num_interation)

    num_iter = 0

    current_train_set = train_intents[0]
    for i in index:
        if num_iter % 50000 == 0 and num_iter > 0:
            print("Change intent to", int(num_iter/50000), "group id", group_sequence[int(num_iter / 50000)])
            all_result = ranker.get_all_query_result_list(current_train_set)
            ndcg = evl_tool.average_ndcg_at_k(current_train_set, all_result, 10)
            ndcg_scores[0].append(ndcg)
            current_train_set = train_intents[group_sequence[int(num_iter / 50000)]]

        qid = query_set[i]
        result_list, scores = ranker.get_query_result_list(current_train_set, qid)

        clicked_doc, click_label, _ = click_model.simulate(qid, result_list, current_train_set)

        ranker.update_to_clicks(click_label, result_list, scores)

        if num_iter % 1000 == 0:
            all_result = ranker.get_all_query_result_list(current_train_set)
            ndcg = evl_tool.average_ndcg_at_k(current_train_set, all_result, 10)
            ndcg_scores[0].append(ndcg)

            for intent in range(len(train_intents)):
                all_result = ranker.get_all_query_result_list(train_intents[intent])
                ndcg = evl_tool.average_ndcg_at_k(train_intents[intent], all_result, 10)
                ndcg_scores[intent + 1].append(ndcg)

        cndcg = evl_tool.query_ndcg_at_k(current_train_set, result_list, qid, 10)
        cndcg_scores.append(cndcg)
        # print(num_iter, ndcg)
        num_iter += 1

    return ndcg_scores, cndcg_scores


def job(model_type, Learning_rate, NUM_INTERACTION, f, train_set, intent_paths, output_fold, num_groups, group_sequence):
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
        datasets = get_groups_dataset(train_set, intent_paths, num_groups=num_groups)
        ranker = PDGDNeuralRanker(FEATURE_SIZE, Learning_rate, [64])

        print("PDGD intent change {} fold{} run{} start!".format(model_type, f, r))
        ndcg_scores, cndcg_scores = run(datasets, ranker, NUM_INTERACTION, cm, group_sequence)

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

        print("PDGD intent change {} run{} finish!".format(model_type, r))
        print()


if __name__ == "__main__":

    FEATURE_SIZE = 105
    NUM_INTERACTION = 200000
    # CHANGE_PER = 500000
    click_models = ["informational", "navigational", "perfect"]
    # click_models = ["perfect"]
    Learning_rate = 0.01
    num_groups = 4
    group_sequence = [0, 1, 2, 3]


    dataset_path = "datasets/clueweb09_intent_change.txt"
    intent_path = "intents"
    output_fold = "results/SDBN/deepPDGD/abrupt_group_change_50k"

    train_set = LetorDataset(dataset_path, FEATURE_SIZE, query_level_norm=True, binary_label=True)

    intent_paths = ["{}/1.txt".format(intent_path),
                    "{}/2.txt".format(intent_path),
                    "{}/3.txt".format(intent_path),
                    "{}/4.txt".format(intent_path)]

    for click_model in click_models:
        mp.Process(target=job, args=(click_model,
                                     Learning_rate,
                                     NUM_INTERACTION,
                                     1,
                                     train_set,
                                     intent_paths,
                                     output_fold,
                                     num_groups,
                                     group_sequence)).start()
