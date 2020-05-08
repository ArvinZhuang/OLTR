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


def run(train_set, test_set, ranker, num_interation, click_model):
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
            print("Change intent to", int(num_inter/10000))
            train_set.update_relevance_label(intents[int(num_inter/10000)])

        qid = query_set[i]
        result_list, scores = ranker.get_query_result_list(train_set, qid)

        clicked_doc, click_label, _ = click_model.simulate(qid, result_list, train_set)

        ranker.update_to_clicks(click_label, result_list, scores, train_set.get_all_features_by_query(qid))

        all_result = ranker.get_all_query_result_list(train_set)
        ndcg = evl_tool.average_ndcg_at_k(train_set, all_result, 10)
        cndcg = evl_tool.query_ndcg_at_k(train_set, result_list, qid, 10)

        # print(num_inter, ndcg)

        ndcg_scores.append(ndcg)
        cndcg_scores.append(cndcg)
        num_inter += 1

    return ndcg_scores, cndcg_scores


def job(model_type, f, train_set, test_set, tau, output_fold):
    if model_type == "perfect":
        pc = [0.0, 1.0]
        cm = PBM(pc, 0)
    elif model_type == "navigational":
        pc = [0.05, 0.95]
        cm = PBM(pc, 1)
    elif model_type == "informational":
        pc = [0.3, 0.7]
        cm = PBM(pc, 1)


    for r in tqdm(range(1, 21), desc='PDGD intent change finised run'):
        # np.random.seed(r)
        ranker = PDGDLinearRanker(FEATURE_SIZE, Learning_rate, tau)

        ndcg_scores, cndcg_scores = run(train_set, test_set, ranker, NUM_INTERACTION, cm)
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
    # click_models = ["informational"]
    Learning_rate = 0.1
    # dataset_fold = "../datasets/MSLR-WEB10K"
    # dataset_fold = "../datasets/2007_mq_dataset"
    dataset_fold = "../datasets/clueweb09"
    output_fold = "results/PDGD"
    # taus = [0.1, 0.5, 1.0, 5.0, 10.0]
    taus = [1]
    # for 5 folds
    for f in range(1, 2):
        # training_path = "{}/Fold{}/train.txt".format(dataset_fold, f)
        # test_path = "{}/Fold{}/test.txt".format(dataset_fold, f)
        training_path = "{}/ClueWeb09-TREC-LTR.txt".format(dataset_fold)
        test_path = "{}/ClueWeb09-TREC-LTR.txt".format(dataset_fold)
        train_set = LetorDataset(training_path, FEATURE_SIZE, query_level_norm=True, binary_label=True)
        test_set = LetorDataset(test_path, FEATURE_SIZE, query_level_norm=True, binary_label=True)

        # for 3 click_models
        for click_model in tqdm(click_models, desc='PDGD Click Model'):
            for tau in taus:
                mp.Process(target=job, args=(click_model, f, train_set, test_set, tau, output_fold)).start()
