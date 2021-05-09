import sys
sys.path.append('../')
from dataset.LetorDataset import LetorDataset
from ranker.PDGDLinearRanker import PDGDLinearRanker
from clickModel.SDBN import SDBN
from utils.utility import get_groups_dataset
from utils import evl_tool
import numpy as np
import multiprocessing as mp
import pickle
import os
import random


def run(train_sets, ranker, num_interation, click_model):

    ndcg_scores = []
    cndcg_scores = []

    query_set = train_sets[0].get_all_querys()
    index = np.random.randint(query_set.shape[0], size=num_interation)
    intents_probs = [0.25,0.25,0.25,0.25]
    num_iter = 0

    for i in index:
        current_intent = np.random.choice(4, 1, p=intents_probs)[0]
        current_train_set = train_sets[current_intent]

        qid = query_set[i]
        result_list, scores = ranker.get_query_result_list(current_train_set, qid)

        clicked_doc, click_label, _ = click_model.simulate(qid, result_list, current_train_set)

        ranker.update_to_clicks(click_label, result_list, scores, current_train_set.get_all_features_by_query(qid))

        if num_iter % 1000 == 0:
            over_all_ndcg = []
            for intent in range(4):
                all_result = ranker.get_all_query_result_list(train_sets[intent])
                ndcg = evl_tool.average_ndcg_at_k(train_sets[intent], all_result, 10)
                over_all_ndcg.append(ndcg)
            over_all_ndcg = np.array(over_all_ndcg)
            over_all_ndcg = np.sum(over_all_ndcg * intents_probs)
            ndcg_scores.append(over_all_ndcg)

        cndcg = evl_tool.query_ndcg_at_k(current_train_set, result_list, qid, 10)
        cndcg_scores.append(cndcg)
        # print(num_iter, ndcg)
        num_iter += 1

    return ndcg_scores, cndcg_scores


def job(model_type, Learning_rate, NUM_INTERACTION, f, train_set, intent_paths, output_fold, num_groups):
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

        ranker = PDGDLinearRanker(FEATURE_SIZE, Learning_rate)

        print("PDGD mixed intent {} run{} start!".format(model_type, r))
        ndcg_scores, cndcg_scores = run(datasets, ranker, NUM_INTERACTION, cm)

        os.makedirs(os.path.dirname("{}/mixed_group/fold{}/".format(output_fold, f)), exist_ok=True)
        with open(
                "{}/mixed_group/fold{}/{}_run{}_ndcg.txt".format(output_fold, f, model_type, r),
                "wb") as fp:
            pickle.dump(ndcg_scores, fp)
        with open(
                "{}/mixed_group/fold{}/{}_run{}_cndcg.txt".format(output_fold, f, model_type, r),
                "wb") as fp:
            pickle.dump(cndcg_scores, fp)

        print("PDGD mixed intent {} run{} finished!".format(model_type, r))
        print()


if __name__ == "__main__":
    FEATURE_SIZE = 105
    NUM_INTERACTION = 2000000
    click_models = ["informational", "navigational", "perfect"]
    # click_models = ["perfect"]
    Learning_rate = 0.1

    num_groups = 4

    dataset_path = "datasets/clueweb09_intent_change.txt"
    intent_path = "intents"
    output_fold = "results/SDBN/PDGD/group_mixed_2m"

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
                                     num_groups)).start()
