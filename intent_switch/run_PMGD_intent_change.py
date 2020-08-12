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
import copy
from tqdm import tqdm

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


def run(train_intents, test_intents, ranker, num_interation, click_model, num_rankers):

    ndcg_scores = [[], [], [], [], []]
    cndcg_scores = []


    query_set = train_set.get_all_querys()
    index = np.random.randint(query_set.shape[0], size=num_interation)
    num_iter = 0

    current_train_set = train_intents[0]
    current_test_set = test_intents[0]

    for i in index:
        if num_iter % 10000 == 0 and num_iter > 0:
            # print("Change intent to", int(num_iter/10000))
            all_result = ranker.get_all_query_result_list(current_test_set, ranker.get_current_weights())
            ndcg = evl_tool.average_ndcg_at_k(current_test_set, all_result, 10)
            ndcg_scores[0].append(ndcg)

            current_train_set = train_intents[int(num_iter / 10000)]
            current_test_set = test_intents[int(num_iter / 10000)]

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

        if num_iter % 100 == 0:
            all_result = ranker.get_all_query_result_list(current_test_set, ranker.get_current_weights())
            ndcg = evl_tool.average_ndcg_at_k(current_test_set, all_result, 10)
            ndcg_scores[0].append(ndcg)

            for intent in range(4):
                all_result = ranker.get_all_query_result_list(test_intents[intent], ranker.get_current_weights())
                ndcg = evl_tool.average_ndcg_at_k(test_intents[intent], all_result, 10)
                ndcg_scores[intent + 1].append(ndcg)

        cndcg = evl_tool.query_ndcg_at_k(current_train_set, inter_list, qid, 10)
        cndcg_scores.append(cndcg)

        num_iter += 1

    return ndcg_scores, cndcg_scores


def job(model_type, f, train_intents, test_intents, delta, alpha, FEATURE_SIZE, num_rankers, output_fold):
    if model_type == "perfect":
        pc = [0.0, 1.0]
        cm = PBM(pc, 0)
    elif model_type == "navigational":
        pc = [0.05, 0.95]
        cm = PBM(pc, 1)
    elif model_type == "informational":
        pc = [0.3, 0.7]
        cm = PBM(pc, 1)


    for r in range(1, 16):
        # np.random.seed(r)
        ranker = ProbabilisticRanker(delta, alpha, FEATURE_SIZE)
        print("PMGD intent change {} fold{} run{} start!".format(model_type, f, r))
        ndcg_scores, cndcg_scores = run(train_intents, test_intents, ranker, NUM_INTERACTION, cm, num_rankers)

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


if __name__ == "__main__":
    FEATURE_SIZE = 105
    NUM_INTERACTION = 40000
    # click_models = ["informational", "navigational", "perfect"]
    click_models = ["informational", "navigational", "perfect"]
    # taus = [0.1, 0.5, 1.0, 5.0, 10.0]
    alpha = 0.01
    delta = 1
    num_rankers = 49

    dataset_fold = "datasets/intent_change_mine"
    intent_path = "intents"
    output_fold = "results/SDBN/PMGD/intent_change"

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

        train_intents = [train_set1, train_set2, train_set3, train_set4]
        test_intents = [test_set1, test_set2, test_set3, test_set4]
        # for 3 click_models
        for click_model in click_models:
            p = mp.Process(target=job, args=(click_model, f, train_intents, test_intents,
                                             delta, alpha, FEATURE_SIZE, num_rankers, output_fold))
            p.start()
