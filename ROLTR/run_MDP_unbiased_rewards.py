from dataset import LetorDataset
import numpy as np
from clickModel.PBM import PBM
from ranker.MDPRanker import MDPRanker
from utils import evl_tool
from utils.utility import GetReturn_DCG
import multiprocessing as mp
import pickle


def run(train_set, test_set, ranker, num_interation, click_model):
    ndcg_scores = []
    cndcg_scores = []
    query_set = train_set.get_all_querys()
    index = np.random.randint(query_set.shape[0], size=num_interation)
    num_iter = 0
    for i in index:
        qid = query_set[i]

        result_list = ranker.get_query_result_list(train_set, qid)
        clicked_doces, click_labels, propensities = click_model.simulate(qid, result_list, train_set)


        if len(clicked_doces) == 0:
            all_result = ranker.get_all_query_result_list(test_set)
            ndcg = evl_tool.average_ndcg_at_k(test_set, all_result, 10)
            cndcg = evl_tool.query_ndcg_at_k(train_set, result_list, qid, 10)

            ndcg_scores.append(ndcg)
            cndcg_scores.append(cndcg)
            num_iter += 1
            continue

        rewards = GetReturn_DCG(click_labels, propensities)

        # ranker.record_episode(qid, result_list, rewards)

        ranker.TFupdate(qid, result_list, rewards, train_set)

        all_result = ranker.get_all_query_result_list(test_set)
        ndcg = evl_tool.average_ndcg_at_k(test_set, all_result, 10)
        cndcg = evl_tool.query_ndcg_at_k(train_set, result_list, qid, 10)
        # print(num_iter, ndcg)
        ndcg_scores.append(ndcg)
        cndcg_scores.append(cndcg)
        num_iter += 1
    return ndcg_scores, cndcg_scores

def job(model_type, f, train_set, test_set, num_features, output_fold):

    if model_type == "perfect":
        pc = [0.0, 0.5, 1.0]
        ps = [0.0, 0.0, 0.0]
    elif model_type == "navigational":
        pc = [0.05, 0.5, 0.95]
        ps = [0.2, 0.5, 0.9]
    elif model_type == "informational":
        pc = [0.4, 0.7, 0.9]
        ps = [0.1, 0.3, 0.5]
    # if model_type == "perfect":
    #     pc = [0.0, 0.2, 0.4, 0.8, 1.0]
    #     ps = [0.0, 0.0, 0.0, 0.0, 0.0]
    # elif model_type == "navigational":
    #     pc = [0.05, 0.3, 0.5, 0.7, 0.95]
    #     ps = [0.2, 0.3, 0.5, 0.7, 0.9]
    # elif model_type == "informational":
    #     pc = [0.4, 0.6, 0.7, 0.8, 0.9]
    #     ps = [0.1, 0.2, 0.3, 0.4, 0.5]
    cm = PBM(pc, 1)

    for r in range(1, 26):
        # np.random.seed(r)
        ranker = MDPRanker(256, num_features, 0.01)
        print("MDP unbiased rewards, mq2007 fold{} {} run{} start!".format(f, model_type, r))
        ndcg_scores, cndcg_scores = run(train_set, test_set, ranker, NUM_INTERACTION, cm)
        with open(
                "{}/fold{}/{}_run{}_ndcg.txt".format(output_fold, f, model_type, r),
                "wb") as fp:
            pickle.dump(ndcg_scores, fp)
        with open(
                "{}/fold{}/{}_run{}_cndcg.txt".format(output_fold, f, model_type, r),
                "wb") as fp:
            pickle.dump(cndcg_scores, fp)

        print("MDP unbiased rewards, mq2007 fold{} {} run{} finished!".format(f, model_type, r))


if __name__ == "__main__":

    FEATURE_SIZE = 64
    NUM_INTERACTION = 10000
    click_models = ["informational", "navigational", "perfect"]
    # click_models = ["perfect"]

    # dataset_fold = "../datasets/MSLR-WEB10K"
    dataset_fold = "../datasets/2007_mq_dataset"
    output_fold = "results/mq2007/MDP_unbiased"

    # for 5 folds
    for f in range(1, 6):
        training_path = "{}/Fold{}/train.txt".format(dataset_fold, f)
        test_path = "{}/Fold{}/test.txt".format(dataset_fold, f)
        train_set = LetorDataset(training_path, FEATURE_SIZE, query_level_norm=False)
        test_set = LetorDataset(test_path, FEATURE_SIZE, query_level_norm=False)

        processors = []
        # for 3 click_models
        for click_model in click_models:
            p = mp.Process(target=job, args=(click_model, f, train_set, test_set, FEATURE_SIZE, output_fold))
            p.start()
            processors.append(p)
        for p in processors:
            p.join()