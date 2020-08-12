import sys
sys.path.append('../')
from dataset.LetorDataset import LetorDataset
from ranker.PDGDLinearRanker import PDGDLinearRanker
from clickModel.SDBN import SDBN
from clickModel.PBM import PBM
from utils import evl_tool
import numpy as np
import multiprocessing as mp
import pickle
import copy
import os
import random


def read_intent_qrel(path: str):
    # q-d pair dictionary
    qrel_dic = {}

    with open(path, 'r') as f:
        for line in f:
            qid, _, docid, rel = line.strip().split()
            if qid in qrel_dic.keys():
                qrel_dic[qid][docid] = int(rel)
            else:
                qrel_dic[qid] = {docid: int(rel)}
    return qrel_dic


def get_intent_dataset(train_set, test_set, intent_path):
    new_train_set = copy.deepcopy(train_set)
    new_test_set = copy.deepcopy(test_set)
    qrel_dic = read_intent_qrel(intent_path)
    new_train_set.update_relevance_label(qrel_dic)
    new_test_set.update_relevance_label(qrel_dic)
    return new_train_set, new_test_set


def run(train_intents, test_intents, ranker, num_interation, click_model):

    ndcg_scores = [[], [], [], [], []]
    cndcg_scores = []

    query_set = train_intents[0].get_all_querys()
    index = np.random.randint(query_set.shape[0], size=num_interation)
    num_iter = 0

    current_train_set = train_intents[0]
    current_test_set = test_intents[0]
    for i in index:
        if num_iter % 10000 == 0 and num_iter > 0:
            # print("Change intent to", int(num_iter/10000))
            all_result = ranker.get_all_query_result_list(current_test_set)
            ndcg = evl_tool.average_ndcg_at_k(current_test_set, all_result, 10)
            ndcg_scores[0].append(ndcg)

            current_train_set = train_intents[int(num_iter/10000)]
            current_test_set = test_intents[int(num_iter / 10000)]

        qid = query_set[i]
        result_list, scores = ranker.get_query_result_list(current_train_set, qid)

        clicked_doc, click_label, _ = click_model.simulate(qid, result_list, current_train_set)

        ranker.update_to_clicks(click_label, result_list, scores, current_train_set.get_all_features_by_query(qid))

        if num_iter % 100 == 0:
            all_result = ranker.get_all_query_result_list(current_test_set)
            ndcg = evl_tool.average_ndcg_at_k(current_test_set, all_result, 10)
            ndcg_scores[0].append(ndcg)

            for intent in range(4):
                all_result = ranker.get_all_query_result_list(test_intents[intent])
                ndcg = evl_tool.average_ndcg_at_k(test_intents[intent], all_result, 10)
                ndcg_scores[intent+1].append(ndcg)

        cndcg = evl_tool.query_ndcg_at_k(current_train_set, result_list, qid, 10)
        cndcg_scores.append(cndcg)
        # print(num_iter, ndcg)
        num_iter += 1

    return ndcg_scores, cndcg_scores


def job(model_type, Learning_rate, NUM_INTERACTION, f, train_intents, test_intents, output_fold):
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

    for r in range(1, 16):
        # np.random.seed(r)
        ranker = PDGDLinearRanker(FEATURE_SIZE, Learning_rate)

        print("PDGD intent change {} fold{} run{} start!".format(model_type, f, r))
        ndcg_scores, cndcg_scores = run(train_intents, test_intents, ranker, NUM_INTERACTION, cm)

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
    NUM_INTERACTION = 40000
    click_models = ["informational", "navigational", "perfect"]
    # click_models = ["navigational"]
    Learning_rate = 0.1

    dataset_fold = "datasets/intent_change_mine"
    intent_path = "intents"
    output_fold = "results/SDBN/PDGD/intent_change_4321"

    # for 5 folds
    for f in range(1, 6):
        training_path = "{}/Fold{}/train.txt".format(dataset_fold, f)
        test_path = "{}/Fold{}/test.txt".format(dataset_fold, f)
        # test_path = "{}/Fold{}/clueweb09_intent_change.txt".format(dataset_fold, f)
        train_set = LetorDataset(training_path, FEATURE_SIZE, query_level_norm=True, binary_label=True)
        test_set = LetorDataset(test_path, FEATURE_SIZE, query_level_norm=True, binary_label=True)

        train_set1, test_set1 = get_intent_dataset(train_set, test_set, "{}/1.txt".format(intent_path))
        train_set2, test_set2 = get_intent_dataset(train_set, test_set, "{}/2.txt".format(intent_path))
        train_set3, test_set3 = get_intent_dataset(train_set, test_set, "{}/3.txt".format(intent_path))
        train_set4, test_set4 = get_intent_dataset(train_set, test_set, "{}/4.txt".format(intent_path))

        train_intents = [train_set4, train_set3, train_set2, train_set1]
        test_intents = [test_set4, test_set3, test_set2, test_set1]

        for click_model in click_models:
            mp.Process(target=job, args=(click_model,
                                             Learning_rate,
                                             NUM_INTERACTION,
                                             f,
                                             train_intents,
                                             test_intents,
                                             output_fold)).start()
