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
from tqdm import tqdm


def run(train_set, test_set, ranker, num_interation, click_model, batch_size):
    ndcg_scores = []
    cndcg_scores = []
    cmrr_scores = []
    query_set = train_set.get_all_querys()
    index = np.random.randint(query_set.shape[0], size=num_interation)
    num_iter = 0
    gradients = np.zeros(train_set._feature_size)
    cmrr = 0
    cndcg = 0
    for i in tqdm(index):
        num_iter += 1

        qid = query_set[i]

        result_list, scores = ranker.get_query_result_list(train_set, qid)

        clicked_doc, click_label, _ = click_model.simulate(qid, result_list, train_set)
        gradients += ranker.update_to_clicks(click_label, result_list, scores, train_set.get_all_features_by_query(qid), return_gradients=True)

        cmrr += evl_tool.online_mrr_at_k(click_label, 10)
        cndcg += evl_tool.query_ndcg_at_k(train_set, result_list, qid, 10)
        if num_iter % batch_size == 0:
            cmrr_scores.append(cmrr/batch_size)
            cndcg_scores.append(cndcg/batch_size)
            cmrr = 0
            cndcg = 0

            # gradients = gradients/batch_size
            ranker.update_to_gradients(gradients)
            gradients = np.zeros(train_set._feature_size)

            all_result = ranker.get_all_query_result_list(test_set)
            ndcg = evl_tool.average_ndcg_at_k(test_set, all_result, 10)
            ndcg_scores.append(ndcg)
            # print(ndcg)

        final_weights = ranker.get_current_weights()
    return ndcg_scores, cndcg_scores, cmrr_scores, final_weights


def job(model_type, f, train_set, test_set, output_fold, batch_size, pc, ps):
    cm = SDBN(pc, ps)

    for r in range(1, 2):
        # np.random.seed(r)
        ranker = PDGDLinearRanker(FEATURE_SIZE, Learning_rate)
        print("PDGD fold{} {} run{} start!".format(f, model_type, r))
        ndcg_scores, cndcg_scores, cmrr_scores, final_weights = run(train_set, test_set, ranker, NUM_INTERACTION, cm, batch_size)
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
                "{}/fold{}/{}_run{}_cmrr.txt".format(output_fold, f, model_type, r),
                "wb") as fp:
            pickle.dump(cmrr_scores, fp)
        with open(
                "{}/fold{}/{}_run{}_weights.txt".format(output_fold, f, model_type, r),
                "wb") as fp:
            pickle.dump(final_weights, fp)
        print("PDGD fold{} {} run{} finished!".format(f, model_type, r))


if __name__ == "__main__":
    NUM_INTERACTION = 2000000
    click_models = ["informational", "navigational", "perfect"]
    Learning_rate = 0.1
    batch_sizes = [8000, 4000, 200]

    mslr10k_fold = "../datasets/MSLR10K"
    mslr10k_output = "../results/FOLTR/PDGD/MSLR10K/MSLR10K_batch_update_size{}_grad_add"
    mq2007_fold = "../datasets/mq2007"
    mq2007_output = "../results/FOLTR/PDGD/mq2008/MQ2008_batch_update_size{}_grad_add"
    mq2008_fold = "../datasets/mq2008"
    mq2008_output = "../results/FOLTR/PDGD/mq2008/MQ2008_batch_update_size{}_grad_add"
    Yahoo_fold = "../datasets/Yahoo"
    Yahoo_output = "../results/FOLTR/PDGD/yahoo/yahoo_batch_update_size{}_grad_add"

    for batch_size in batch_sizes:
        paths = [
                # (mslr10k_fold, mslr10k_output.format(batch_size)),
                # (mq2007_fold, mq2007_output.format(batch_size)),
                (mq2008_fold, mq2008_output.format(batch_size)),
                # (Yahoo_fold, Yahoo_output)
        ]
        for path in paths:
            dataset_fold = path[0]
            output_fold = path[1]

            processors = []
            for click_model in click_models:
                if dataset_fold == "../datasets/MSLR10K":
                    FEATURE_SIZE = 136
                    norm = True
                    fold_range = range(1, 6)
                    if click_model == "perfect":
                        pc = [0.0, 0.2, 0.4, 0.8, 1.0]
                        ps = [0.0, 0.0, 0.0, 0.0, 0.0]
                    elif click_model == "navigational":
                        pc = [0.05, 0.3, 0.5, 0.7, 0.95]
                        ps = [0.2, 0.3, 0.5, 0.7, 0.9]
                    elif click_model == "informational":
                        pc = [0.4, 0.6, 0.7, 0.8, 0.9]
                        ps = [0.1, 0.2, 0.3, 0.4, 0.5]
                elif dataset_fold == "../datasets/mq2007" or dataset_fold == "../datasets/mq2008":
                    FEATURE_SIZE = 46
                    norm = False
                    fold_range = range(1, 6)
                    if click_model == "perfect":
                        pc = [0.0, 0.5, 1.0]
                        ps = [0.0, 0.0, 0.0]
                    elif click_model == "navigational":
                        pc = [0.05, 0.5, 0.95]
                        ps = [0.2, 0.5, 0.9]
                    elif click_model == "informational":
                        pc = [0.4, 0.7, 0.9]
                        ps = [0.1, 0.3, 0.5]
                elif dataset_fold == "../datasets/Yahoo":
                    FEATURE_SIZE = 700
                    norm = False
                    fold_range = range(1, 2)
                    if click_model == "perfect":
                        pc = [0.0, 0.2, 0.4, 0.8, 1.0]
                        ps = [0.0, 0.0, 0.0, 0.0, 0.0]
                    elif click_model == "navigational":
                        pc = [0.05, 0.3, 0.5, 0.7, 0.95]
                        ps = [0.2, 0.3, 0.5, 0.7, 0.9]
                    elif click_model == "informational":
                        pc = [0.4, 0.6, 0.7, 0.8, 0.9]
                        ps = [0.1, 0.2, 0.3, 0.4, 0.5]

                for f in fold_range:
                    training_path = "{}/Fold{}/train.txt".format(dataset_fold, f)
                    test_path = "{}/Fold{}/test.txt".format(dataset_fold, f)
                    train_set = LetorDataset(training_path, FEATURE_SIZE, query_level_norm=norm, cache_root="../datasets/cache")
                    test_set = LetorDataset(test_path, FEATURE_SIZE, query_level_norm=norm, cache_root="../datasets/cache")

                    print(dataset_fold, click_model, f, batch_size)
                    p = mp.Process(target=job, args=(click_model, f, train_set, test_set, output_fold, batch_size, pc, ps))
                    p.start()
                    processors.append(p)
            for p in processors:
                p.join()