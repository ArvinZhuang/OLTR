import sys
sys.path.append('../')
from dataset.LetorDataset import LetorDataset
from ranker.COLTRLinearRanker import COLTRLinearRanker
from clickModel.SDBN import SDBN
from utils import evl_tool
import numpy as np
import multiprocessing as mp
import pickle
import copy
from utils import utility


def run(train_set, test_set, ranker, num_interation, click_model, num_rankers):

    ndcg_scores = []
    cndcg_scores = []
    query_set = train_set.get_all_querys()
    index = np.random.randint(query_set.shape[0], size=num_interation)

    num_interation = 0

    has_winners = False
    correct = 0
    wrong = 0
    for i in index:
        num_interation += 1
        qid = query_set[i]

        result_list = ranker.get_query_result_list(train_set, qid)

        clicked_doc, click_label, _ = click_model.simulate(qid, result_list, train_set)

        # if no clicks, skip.
        if len(clicked_doc) == 0:
            all_result = ranker.get_all_query_result_list(test_set)
            ndcg = evl_tool.average_ndcg_at_k(test_set, all_result, 10)
            cndcg = evl_tool.query_ndcg_at_k(train_set, result_list, qid, 10)

            ndcg_scores.append(ndcg)
            cndcg_scores.append(cndcg)
            continue

        # flip click label. exp: [1,0,1,0,0] -> [0,1,0,0,0]
        last_click = np.where(click_label == 1)[0][-1]
        click_label[:last_click + 1] = 1 - click_label[:last_click + 1]

        # bandit record
        record = (qid, result_list, click_label, ranker.get_current_weights())

        unit_vectors = ranker.sample_unit_vectors(num_rankers)
        canditate_rankers = ranker.sample_canditate_rankers(unit_vectors)  # canditate_rankers are ranker weights, not ranker class

        if has_winners:
            canditate_rankers = np.vstack((winner_rankers, canditate_rankers))


        winner_rankers = ranker.infer_winners(canditate_rankers[:num_rankers], record)

        # """ This part of code is used to test correctness of counterfactual evaluation
        if winner_rankers is not None:
            SERP = utility.get_query_result_list(ranker.get_current_weights(), train_set, qid)
            current_ndcg = evl_tool.query_ndcg_at_k(train_set, SERP, qid, 10)
            for weights in canditate_rankers[winner_rankers - 1]:
                canditate_SERP = utility.get_query_result_list(weights, train_set, qid)
                canditate_SERP_ndcg = evl_tool.query_ndcg_at_k(train_set, canditate_SERP, qid, 10)

                if canditate_SERP_ndcg >= current_ndcg:
                    correct += 1
                else:
                    wrong += 1
            print(correct, wrong, correct / (correct + wrong))
        # """

        if winner_rankers is not None:
            gradient = np.sum(unit_vectors[winner_rankers - 1], axis=0) / winner_rankers.shape[0]
            ranker.update(gradient)
            winner_rankers = canditate_rankers[winner_rankers - 1]
            has_winners = True
        else:
            has_winners = False

        all_result = ranker.get_all_query_result_list(test_set)
        ndcg = evl_tool.average_ndcg_at_k(test_set, all_result, 10)
        cndcg = evl_tool.query_ndcg_at_k(train_set, result_list, qid, 10)

        ndcg_scores.append(ndcg)
        cndcg_scores.append(cndcg)
        final_weight = ranker.get_current_weights()
        # print(num_interation, ndcg, cndcg)

    return ndcg_scores, cndcg_scores, final_weight


def job(model_type, f, train_set, test_set, tau, step_size, gamma, num_rankers, learning_rate_decay, output_fold):
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

    cm = SDBN(pc, ps)

    for r in range(1, 26):
        # np.random.seed(r)
        # new a new ranker
        ranker = COLTRLinearRanker(FEATURE_SIZE, Learning_rate, step_size, tau, gamma, learning_rate_decay=learning_rate_decay)

        print("COTLR {} tau{} fold{} {} run{} start!".format(output_fold, tau, f, model_type, r))
        ndcg_scores, cndcg_scores, final_weight = run(train_set, test_set, ranker, NUM_INTERACTION, cm, num_rankers)
        # with open(
        #         "{}/fold{}/{}_tau{}_run{}_ndcg.txt".format(output_fold, f, model_type, tau, r),
        #         "wb") as fp:
        #     pickle.dump(ndcg_scores, fp)
        # with open(
        #         "{}/fold{}/{}_tau{}_run{}_cndcg.txt".format(output_fold, f, model_type, tau, r),
        #         "wb") as fp:
        #     pickle.dump(cndcg_scores, fp)
        # with open(
        #         "{}/fold{}/{}_tau{}_run{}_final_weight.txt".format(output_fold, f, model_type, tau, r),
        #         "wb") as fp:
        #     pickle.dump(final_weight, fp)
        # print("COTLR {} tau{} fold{} {} run{} finished!".format(output_fold, tau, f, model_type, r))

        utility.send_progress(model_type, r, 25, "final ndcg {}".format(ndcg_scores[-1]))


if __name__ == "__main__":

    FEATURE_SIZE = 46
    NUM_INTERACTION = 10000
    # click_models = ["informational", "navigational", "perfect"]
    click_models = ["informational"]
    Learning_rate = 0.1
    # dataset_fold = "../datasets/MSLR-WEB10K"
    # output_fold = "../results/COLTR/MSLR-WEB10K"
    dataset_fold = "../datasets/2007_mq_dataset"
    output_fold = "../results/COLTR/mq2007"

    num_rankers = 499
    tau = 1
    gamma = 1
    learning_rate_decay = 0.99966
    step_size = 1

    # for 5 folds
    for f in range(1, 6):
        training_path = "{}/Fold{}/train.txt".format(dataset_fold, f)
        test_path = "{}/Fold{}/test.txt".format(dataset_fold, f)
        print("loading fold{} training set and test set....".format(f))
        train_set = LetorDataset(training_path, FEATURE_SIZE, query_level_norm=True)
        test_set = LetorDataset(test_path, FEATURE_SIZE, query_level_norm=True)

        processors = []
        # for 3 click_models
        for click_model in click_models:
            p = mp.Process(target=job, args=(click_model, f, train_set, test_set,
                                         tau, step_size, gamma, num_rankers, learning_rate_decay, output_fold))
            p.start()
            processors.append(p)
        for p in processors:
            p.join()

        utility.send_progress("COLTR", f, 5, "MSLR-WEB10K")


