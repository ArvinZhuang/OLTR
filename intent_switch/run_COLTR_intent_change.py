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
        result_list = ranker.get_query_result_list(train_set, qid)

        clicked_doc, click_label, _ = click_model.simulate(qid, result_list, train_set)

        # if no clicks, skip.
        if len(clicked_doc) == 0:
            all_result = ranker.get_all_query_result_list(train_set)
            ndcg = evl_tool.average_ndcg_at_k(train_set, all_result, 10)
            cndcg = evl_tool.query_ndcg_at_k(train_set, result_list, qid, 10)

            ndcg_scores.append(ndcg)
            cndcg_scores.append(cndcg)
            # print(num_inter, ndcg, "continue")
            num_inter += 1
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

        all_result = ranker.get_all_query_result_list(train_set)
        ndcg = evl_tool.average_ndcg_at_k(train_set, all_result, 10)
        cndcg = evl_tool.query_ndcg_at_k(train_set, result_list, qid, 10)

        ndcg_scores.append(ndcg)
        cndcg_scores.append(cndcg)
        # print(num_inter, ndcg)
        num_inter += 1

    return ndcg_scores, cndcg_scores


def job(model_type, f, train_set, test_set, tau, step_size, gamma, num_rankers, learning_rate_decay, output_fold):
    if model_type == "perfect":
        pc = [0.0, 1.0]
        cm = PBM(pc, 0)
    elif model_type == "navigational":
        pc = [0.05, 0.95]
        cm = PBM(pc, 1)
    elif model_type == "informational":
        pc = [0.3, 0.7]
        cm = PBM(pc, 1)


    for r in tqdm(range(1, 21), desc='COLTR intent change finised run'):
        # np.random.seed(r)
        ranker = COLTRLinearRanker(FEATURE_SIZE, Learning_rate, step_size, tau, gamma, learning_rate_decay=learning_rate_decay)

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

    FEATURE_SIZE = 91
    NUM_INTERACTION = 50000
    # click_models = ["informational", "navigational", "perfect"]
    click_models = ["informational", "perfect"]
    # click_models = ["perfect"]
    Learning_rate = 0.05
    # dataset_fold = "../datasets/MSLR-WEB10K"
    # dataset_fold = "../datasets/2007_mq_dataset"
    dataset_fold = "../datasets/clueweb09"
    output_fold = "results/COLTR"
    # taus = [0.1, 0.5, 1.0, 5.0, 10.0]
    num_rankers = 499
    tau = 0.1
    gamma = 1
    learning_rate_decay = 1
    step_size = 1

    # for 5 folds
    for f in range(1, 2):
        # training_path = "{}/Fold{}/train.txt".format(dataset_fold, f)
        # test_path = "{}/Fold{}/test.txt".format(dataset_fold, f)
        training_path = "{}/ClueWeb09-TREC-LTR.txt".format(dataset_fold)
        test_path = "{}/ClueWeb09-TREC-LTR.txt".format(dataset_fold)
        train_set = LetorDataset(training_path, FEATURE_SIZE, query_level_norm=True, binary_label=True)
        test_set = LetorDataset(test_path, FEATURE_SIZE, query_level_norm=True, binary_label=True)

        # for 3 click_models
        for click_model in tqdm(click_models, desc='COLTR'):
            p = mp.Process(target=job, args=(click_model, f, train_set, test_set,
                                             tau, step_size, gamma, num_rankers, learning_rate_decay, output_fold))
            p.start()
