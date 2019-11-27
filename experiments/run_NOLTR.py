import sys
sys.path.append('../')
from dataset.LetorDataset import LetorDataset
from ranker.NeuralRanker import NeuralRanker
from clickModel.SDBN import CascadeClickModel
from utils import evl_tool
import numpy as np
import multiprocessing as mp
import pickle


def run(train_set, test_set, ranker, num_interation, click_model):
    ndcg_scores = []
    cndcg_scores = []
    query_set = train_set.get_all_querys()
    index = np.random.randint(query_set.shape[0], size=num_interation)
    num_interation = 0
    for i in index:
        num_interation += 1
        qid = query_set[i]

        result_list, scores = ranker.get_query_result_list(train_set, qid)

        clicked_doc, click_label, _ = click_model.simulate(qid, result_list, train_set)
        if len(clicked_doc) > 0:
            ranker.update(click_label, result_list, train_set.get_all_features_by_query(qid))

        all_result = ranker.get_all_query_result_list(test_set)
        ndcg = evl_tool.average_ndcg_at_k(test_set, all_result, 10)
        cndcg = evl_tool.query_ndcg_at_k(train_set, result_list, qid, 10)

        ndcg_scores.append(ndcg)
        cndcg_scores.append(cndcg)
        final_weight = ranker.get_current_weights()
        print(num_interation, ndcg, cndcg)

    return ndcg_scores, cndcg_scores, final_weight


def job(model_type, f, train_set, test_set):
    if model_type == "perfect":
        pc = [0.0, 0.5, 1.0]
        ps = [0.0, 0.0, 0.0]
    elif model_type == "navigational":
        pc = [0.05, 0.5, 0.95]
        ps = [0.2, 0.5, 0.9]
    elif model_type == "informational":
        pc = [0.4, 0.7, 0.9]
        ps = [0.1, 0.3, 0.5]

    output_fold = "mq2007"
    cm = CascadeClickModel(pc, ps)

    for r in range(1, 26):
        # np.random.seed(r)
        ranker = NeuralRanker(FEATURE_SIZE, Learning_rate)
        # print("PDGD tau{} fold{} {} run{} start!".format(f, model_type, r))
        ndcg_scores, cndcg_scores, final_weight = run(train_set, test_set, ranker, NUM_INTERACTION, cm)
        # with open(
        #         "../results/exploration/mq2007/PDGD/fold{}/{}_tau{}_run{}_ndcg.txt".format(f, model_type, tau, r),
        #         "wb") as fp:
        #     pickle.dump(ndcg_scores, fp)
        # with open(
        #         "../results/exploration/mq2007/PDGD/fold{}/{}_tau{}_run{}_cndcg.txt".format(f, model_type, tau, r),
        #         "wb") as fp:
        #     pickle.dump(cndcg_scores, fp)
        # with open(
        #         "../results/exploration/mq2007/PDGD/fold{}/{}_tau{}_run{}_final_weight.txt".format(f, model_type, tau, r),
        #         "wb") as fp:
        #     pickle.dump(final_weight, fp)
        # print("PDGD tau{} fold{} {} run{} finished!".format(f, model_type, r))


if __name__ == "__main__":

    FEATURE_SIZE = 46
    NUM_INTERACTION = 10000
    # click_models = ["informational", "navigational", "perfect"]
    click_models = ["informational"]
    Learning_rate = 0.01
    dataset_fold = "../datasets/2007_mq_dataset"
    output_fold = "mq2007"
    # for 5 folds
    for f in range(1, 6):
        training_path = "{}/Fold{}/train.txt".format(dataset_fold, f)
        test_path = "{}/Fold{}/test.txt".format(dataset_fold, f)
        train_set = LetorDataset(training_path, FEATURE_SIZE)
        test_set = LetorDataset(test_path, FEATURE_SIZE)

        # for 3 click_models
        for click_model in click_models:
            mp.Process(target=job, args=(click_model, f, train_set, test_set)).start()
            break
        break