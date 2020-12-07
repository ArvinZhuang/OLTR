import os
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

FEATURE_SIZE = 136
NUM_INTERACTION = 1000000
Learning_rate = 0.1


def run(train_set, test_set, ranker, num_interation, click_model):
    ndcg_scores = []
    cndcg_scores = []
    query_set = train_set.get_all_querys()
    index = np.random.randint(query_set.shape[0], size=num_interation)
    num_iter = 0
    for i in index:
        num_iter += 1

        qid = query_set[i]

        result_list, scores = ranker.get_query_result_list(train_set, qid, random=True)

        clicked_doc, click_label, _ = click_model.simulate(qid, result_list, train_set)

        ranker.update_to_clicks(click_label, result_list, scores, train_set.get_all_features_by_query(qid))

        if num_iter % 1000 == 0 or num_iter == 1:
            all_result = ranker.get_all_query_result_list(test_set)
            ndcg = evl_tool.average_ndcg_at_k(test_set, all_result, 10)
            ndcg_scores.append(ndcg)
            # print(ndcg)
        cndcg = evl_tool.online_mrr_at_k(click_label, 10)
        cndcg_scores.append(cndcg)
        final_weights = ranker.get_current_weights()
    return ndcg_scores, cndcg_scores, final_weights


def job(model_type, f, train_set, test_set, output_fold):
    if model_type == "perfect":
        # pc = [0.0, 0.2, 0.4, 0.8, 1.0]
        # pc = [0.0, 0.5, 1.0]
        pc = [0.0, 1.0]
        # ps = [0.0, 0.0, 0.0, 0.0, 0.0]
        # ps = [0.0, 0.0, 0.0]
        ps = [0.0, 0.0]
    elif model_type == "navigational":
        pc = [0.05, 0.3, 0.5, 0.7, 0.95]
        # pc = [0.05, 0.5, 0.95]
        ps = [0.2, 0.3, 0.5, 0.7, 0.9]
        # ps = [0.2, 0.5, 0.9]
    elif model_type == "informational":
        # pc = [0.4, 0.6, 0.7, 0.8, 0.9]
        # pc = [0.4, 0.7, 0.9]
        pc = [0.1, 0.9]
        # ps = [0.1, 0.2, 0.3, 0.4, 0.5]
        # ps = [0.1, 0.3, 0.5]
        ps = [0.1, 0.5]

    cm = SDBN(pc, ps)

    for r in range(1, 16):
        # np.random.seed(r)
        FEATURE_SIZE = 136
        ranker = PDGDLinearRanker(FEATURE_SIZE, Learning_rate)
        print("PDGD fold{} {} run{} start!".format(f, model_type, r))
        ndcg_scores, cndcg_scores, final_weights = run(train_set, test_set, ranker, NUM_INTERACTION, cm)
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
                "{}/fold{}/{}_run{}_weights.txt".format(output_fold, f, model_type, r),
                "wb") as fp:
            pickle.dump(final_weights, fp)
        print("PDGD fold{} {} run{} finished!".format(f, model_type, r))


if __name__ == "__main__":
    # click_models = ["informational", "navigational", "perfect"]
    click_models = ["perfect", "informational"]
    dataset_fold = "../datasets/MSLR10k"
    output_fold = "../results/exploration/PDGD/MSLR10K/random"
    # for 5 folds

    for f in range(1, 2):
        training_path = "{}/Fold{}/train.txt".format(dataset_fold, f)
        test_path = "{}/Fold{}/test.txt".format(dataset_fold, f)
        train_set = LetorDataset(training_path, FEATURE_SIZE, query_level_norm=True, cache_root="../datasets/cache", binary_label=3)
        test_set = LetorDataset(test_path, FEATURE_SIZE, query_level_norm=True, cache_root="../datasets/cache", binary_label=3)
        for click_model in click_models:
            mp.Process(target=job, args=(click_model, f, train_set, test_set, output_fold)).start()
