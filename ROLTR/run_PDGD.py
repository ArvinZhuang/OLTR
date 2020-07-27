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
import os


def run(train_set, test_set, ranker, num_interation, click_model):
    ndcg_scores = []
    cndcg_scores = []
    query_set = train_set.get_all_querys()
    index = np.random.randint(query_set.shape[0], size=num_interation)
    num_iter = 0
    for i in index:
        qid = query_set[i]

        result_list, scores = ranker.get_query_result_list(train_set, qid)

        clicked_doc, click_label, _ = click_model.simulate(qid, result_list, train_set)

        ranker.update_to_clicks(click_label, result_list, scores, train_set.get_all_features_by_query(qid))


        if num_iter % 1000 == 0:
            all_result = ranker.get_all_query_result_list(test_set)
            ndcg = evl_tool.average_ndcg_at_k(test_set, all_result, 10)
            # print(ndcg)
            ndcg_scores.append(ndcg)
        cndcg = evl_tool.query_ndcg_at_k(train_set, result_list, qid, 10)
        cndcg_scores.append(cndcg)
        final_weight = ranker.get_current_weights()
        num_iter += 1

        # print(num_iter, ndcg)
    return ndcg_scores, cndcg_scores, final_weight


def job(model_type, f, train_set, test_set, tau, output_fold):
    if model_type == "perfect":
        pc = [0.0, 0.2, 0.4, 0.8, 1.0]
        ps = [0.0, 0.0, 0.0, 0.0, 0.0]
    elif model_type == "navigational":
        pc = [0.05, 0.3, 0.5, 0.7, 0.95]
        ps = [0.2, 0.3, 0.5, 0.7, 0.9]
    elif model_type == "informational":
        pc = [0.4, 0.6, 0.7, 0.8, 0.9]
        ps = [0.1, 0.2, 0.3, 0.4, 0.5]
    # if model_type == "perfect":
    #     pc = [0.0, 0.5, 1.0]
    #     ps = [0.0, 0.0, 0.0]
    # elif model_type == "navigational":
    #     pc = [0.05, 0.5, 0.95]
    #     ps = [0.2, 0.5, 0.9]
    # elif model_type == "informational":
    #     pc = [0.4, 0.7, 0.9]
    #     ps = [0.1, 0.3, 0.5]
    cm = PBM(pc, 1)

    for r in range(16, 26):
        # np.random.seed(r)
        ranker = PDGDLinearRanker(FEATURE_SIZE, Learning_rate, tau)
        print("PDGD tau{} fold{} {} run{} start!".format(tau, f, model_type, r))
        ndcg_scores, cndcg_scores, final_weight = run(train_set, test_set, ranker, NUM_INTERACTION, cm)
        os.makedirs(os.path.dirname("{}/fold{}/".format(output_fold, f)),
                    exist_ok=True)  # create directory if not exist
        with open(
                "{}/fold{}/{}_run{}_ndcg.txt".format(output_fold, f, model_type, r),
                "wb") as fp:
            pickle.dump(ndcg_scores, fp)
        with open(
                "{}/fold{}/{}_run{}_cndcg.txt".format(output_fold, f, model_type, r),
                "wb") as fp:
            pickle.dump(cndcg_scores, fp)
        with open(
                "../results/exploration/mq2007/PDGD/fold{}/{}_tau{}_run{}_final_weight.txt".format(f, model_type, tau, r),
                "wb") as fp:
            pickle.dump(final_weight, fp)
        print("PDGD tau{} fold{} {} run{} finished!".format(tau, f, model_type, r))


if __name__ == "__main__":

    FEATURE_SIZE = 220
    NUM_INTERACTION = 100000
    # click_models = ["informational"]
    click_models = ["informational", "perfect"]
    Learning_rate = 0.1
    # dataset_fold = "../datasets/MSLR10K"
    # dataset_fold = "../datasets/2007_mq_dataset"
    output_fold = "results/istella/PDGD"
    # output_fold = "results/mslr10k/long_term_200k/PDGD_eta2"
    # output_fold = "results/mq2007/PDGD"
    # taus = [0.1, 0.5, 1.0, 5.0, 10.0]
    taus = [1]
    # for 5 folds
    for f in range(1, 2):
        # training_path = "{}/Fold{}/train.txt".format(dataset_fold, f)
        # test_path = "{}/Fold{}/test.txt".format(dataset_fold, f)
        training_path = "../datasets/istella/train.txt"
        test_path = "../datasets/istella/test.txt"
        train_set = LetorDataset(training_path, FEATURE_SIZE, query_level_norm=True)
        test_set = LetorDataset(test_path, FEATURE_SIZE, query_level_norm=True)

        processors = []
        # for 3 click_models
        for click_model in click_models:
            for tau in taus:
                p = mp.Process(target=job, args=(click_model, f, train_set, test_set, tau, output_fold))
                p.start()
                processors.append(p)
    for p in processors:
        p.join()
