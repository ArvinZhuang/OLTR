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


def run(train_set, test_set, ranker, num_interation, click_model, num_rankers):
    qrel_dic1 = read_intent_qrel("1.txt")
    qrel_dic2 = read_intent_qrel("2.txt")
    qrel_dic3 = read_intent_qrel("3.txt")
    # qrel_dic4 = read_intent_qrel("4.txt")
    intents = [qrel_dic1, qrel_dic2, qrel_dic3, qrel_dic1, qrel_dic1]

    ndcg_scores = []
    cndcg_scores = []



    query_set = train_set.get_all_querys()
    index = np.random.randint(query_set.shape[0], size=num_interation)
    num_inter = 0
    for i in index:
        if num_inter % 10000 == 0:
            # print("Change intent to", int(num_inter/10000))
            train_set.update_relevance_label(intents[int(num_inter/10000)])

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

        all_result = ranker.get_all_query_result_list(train_set, ranker.get_current_weights())
        ndcg = evl_tool.average_ndcg_at_k(train_set, all_result, 10)
        cndcg = evl_tool.query_ndcg_at_k(train_set, inter_list, qid, 10)
        # print(num_inter, ndcg)

        ndcg_scores.append(ndcg)
        cndcg_scores.append(cndcg)
        num_inter += 1

    return ndcg_scores, cndcg_scores


def job(model_type, f, train_set, test_set, delta, alpha, FEATURE_SIZE, num_rankers, output_fold):
    if model_type == "perfect":
        pc = [0.0, 1.0]
        cm = PBM(pc, 0)
    elif model_type == "navigational":
        pc = [0.05, 0.95]
        cm = PBM(pc, 1)
    elif model_type == "informational":
        pc = [0.3, 0.7]
        cm = PBM(pc, 1)


    for r in tqdm(range(1, 21), desc='PMGD intent change finised run'):
        # np.random.seed(r)
        ranker = ProbabilisticRanker(delta, alpha, FEATURE_SIZE)

        ndcg_scores, cndcg_scores = run(train_set, test_set, ranker, NUM_INTERACTION, cm, num_rankers)
        with open(
                "{}/{}_run{}_ndcg.txt".format(output_fold, model_type, r),
                "wb") as fp:
            pickle.dump(ndcg_scores, fp)
        with open(
                "{}/{}_run{}_cndcg.txt".format(output_fold, model_type, r),
                "wb") as fp:
            pickle.dump(cndcg_scores, fp)
        # with open(
        #         "../results/exploration/mq2007/PDGD/fold{}/{}_tau{}_run{}_final_weight.txt".format(f, model_type, tau, r),
        #         "wb") as fp:
        #     pickle.dump(final_weight, fp)


if __name__ == "__main__":
    FEATURE_SIZE = 105
    NUM_INTERACTION = 50000
    # click_models = ["informational", "navigational", "perfect"]
    click_models = ["informational", "perfect"]

    dataset_fold = "datasets/intent_change_mine"
    output_fold = "results/SDBN/PMDB/intent_change"
    # taus = [0.1, 0.5, 1.0, 5.0, 10.0]
    alpha = 0.01
    delta = 1
    num_rankers = 49

    # for 5 folds
    for f in range(1, 6):
        training_path = "{}/Fold{}/train.txt".format(dataset_fold, f)
        test_path = "{}/Fold{}/test.txt".format(dataset_fold, f)

        train_set = LetorDataset(training_path, FEATURE_SIZE, query_level_norm=True, binary_label=True)
        test_set = LetorDataset(test_path, FEATURE_SIZE, query_level_norm=True, binary_label=True)

        # for 3 click_models
        for click_model in tqdm(click_models, desc='PMGD'):
            p = mp.Process(target=job, args=(click_model, f, train_set, test_set,
                                             delta, alpha, FEATURE_SIZE, num_rankers, output_fold))
            p.start()
