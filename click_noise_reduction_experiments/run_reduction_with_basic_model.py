import sys
sys.path.append('../')
from dataset.LetorDataset import LetorDataset
from ranker.PDGDLinearRanker import PDGDLinearRanker
from clickModel.SDBN import SDBN
from utils import evl_tool
import numpy as np
import multiprocessing as mp
import pickle


def run(train_set, test_set, ranker1, ranker2, num_interation, click_model):
    click_predictor = SDBN()
    ndcg_scores1 = []
    cndcg_scores1 = []
    ndcg_scores2 = []
    cndcg_scores2 = []
    query_set = train_set.get_all_querys()
    np.random.shuffle(query_set)
    index = np.random.randint(query_set.shape[0], size=num_interation)

    pdf = np.random.normal(size=query_set.shape[0])
    e_x = np.exp((pdf - np.max(pdf)) / 0.2)
    probs = e_x / e_x.sum(axis=0)

    querys = np.random.choice(query_set,
                     replace=True,
                     p=probs,
                     size=num_interation)


    num_interaction = 0
    correct = 0
    wrong = 0
    test1 = 0
    test2 = 0
    for qid in querys:
        num_interaction += 1
        # qid = query_set[i]

        result_list1, scores1 = ranker1.get_query_result_list(train_set, qid)
        result_list2, scores2 = ranker2.get_query_result_list(train_set, qid)

        clicked_doc1, click_label1, _ = click_model.simulate(qid, result_list1, train_set)
        clicked_doc2, click_label2, _ = click_model.simulate(qid, result_list2, train_set)
        #
        last_exam = None
        # if len(clicked_doc2) > 0:
        #     last_exam = np.where(click_label2 == 1)[0][-1] + 1
        #
        #     click_predictor.online_training(qid, result_list2, click_label2)
        #     reduce, reduced_index = click_predictor.click_noise_reduce(qid, result_list2, click_label2, 0.5, 20)
        #
        #     if reduce:
        #         for rank in reduced_index:
        #             # print(train_set.get_relevance_label_by_query_and_docid(qid, result_list2[rank]))
        #             if train_set.get_relevance_label_by_query_and_docid(qid, result_list2[rank]) == 0:
        #                 correct += 1
        #             else:
        #                 wrong += 1
        #         # print(correct, wrong)
        clicked_doc_index = 0
        for j in np.where(click_label2 == 1)[0]:
            rel = train_set.get_relevance_label_by_query_and_docid(qid, result_list2[clicked_doc_index])
            if rel == 0:
                click_label2[j] = 0
            clicked_doc_index += 1



        ranker1.update_to_clicks(click_label1, result_list1, scores1, train_set.get_all_features_by_query(qid))
        ranker2.update_to_clicks(click_label2, result_list2, scores2, train_set.get_all_features_by_query(qid), last_exam)



        all_result1 = ranker1.get_all_query_result_list(test_set)
        ndcg1 = evl_tool.average_ndcg_at_k(test_set, all_result1, 10)
        cndcg1 = evl_tool.query_ndcg_at_k(train_set, result_list1, qid, 10)

        all_result2 = ranker2.get_all_query_result_list(test_set)
        ndcg2 = evl_tool.average_ndcg_at_k(test_set, all_result2, 10)
        cndcg2 = evl_tool.query_ndcg_at_k(train_set, result_list2, qid, 10)

        ndcg_scores1.append(ndcg1)
        cndcg_scores1.append(cndcg1)
        ndcg_scores2.append(ndcg2)
        cndcg_scores2.append(cndcg2)
        final_weight1 = ranker1.get_current_weights()
        final_weight2 = ranker2.get_current_weights()

        test1 += ndcg1
        test2 += ndcg2
    print(test1, test2)
    print(np.mean(ndcg_scores1), np.mean(ndcg_scores2))

    return ndcg_scores1, cndcg_scores1, final_weight1, ndcg_scores2, cndcg_scores2, final_weight2


def job(model_type, f, train_set, test_set, tau):
    if model_type == "perfect":
        pc = [0.0, 0.5, 1.0]
        ps = [0.0, 0.0, 0.0]
    elif model_type == "navigational":
        pc = [0.05, 0.5, 0.95]
        ps = [0.2, 0.5, 0.9]
    elif model_type == "informational":
        pc = [0.4, 0.7, 0.9]
        ps = [0.1, 0.3, 0.5]

    cm = SDBN(pc, ps)

    for r in range(1, 26):
        # np.random.seed(r)
        ranker1 = PDGDLinearRanker(FEATURE_SIZE, Learning_rate, tau)
        ranker2 = PDGDLinearRanker(FEATURE_SIZE, Learning_rate, tau)
        print("PDGD tau{} fold{} {} run{} start!".format(tau, f, model_type, r))
        ndcg_scores1, cndcg_scores1, final_weight1, ndcg_scores2, cndcg_scores2, final_weight2 = run(train_set, test_set, ranker1, ranker2, NUM_INTERACTION, cm)
        with open(
                "../results/reduction/mq2007/PDGD/fold{}/{}_ranker{}_run{}_ndcg.txt".format(f, model_type, 1, r),
                "wb") as fp:
            pickle.dump(ndcg_scores1, fp)
        with open(
                "../results/reduction/mq2007/PDGD/fold{}/{}_ranker{}_run{}_cndcg.txt".format(f, model_type, 1, r),
                "wb") as fp:
            pickle.dump(cndcg_scores1, fp)
        with open(
                "../results/reduction/mq2007/PDGD/fold{}/{}_ranker{}_run{}_final_weight.txt".format(f, model_type, 1, r),
                "wb") as fp:
            pickle.dump(final_weight1, fp)

        with open(
                "../results/reduction/mq2007/PDGD/fold{}/{}_ranker{}_run{}_ndcg.txt".format(f, model_type, 2, r),
                "wb") as fp:
            pickle.dump(ndcg_scores2, fp)
        with open(
                "../results/reduction/mq2007/PDGD/fold{}/{}_ranker{}_run{}_cndcg.txt".format(f, model_type, 2, r),
                "wb") as fp:
            pickle.dump(cndcg_scores2, fp)
        with open(
                "../results/reduction/mq2007/PDGD/fold{}/{}_ranker{}_run{}_final_weight.txt".format(f, model_type, 2, r),
                "wb") as fp:
            pickle.dump(final_weight2, fp)
        print("PDGD tau{} fold{} {} run{} finished!".format(tau, f, model_type, r))


if __name__ == "__main__":

    FEATURE_SIZE = 46
    NUM_INTERACTION = 10000
    # click_models = ["informational", "navigational", "perfect"]
    click_model = "informational"
    Learning_rate = 0.1
    dataset_fold = "../datasets/2007_mq_dataset"
    output_fold = "mq2007"
    # taus = [0.1, 0.5, 1.0, 5.0, 10.0]
    tau = 1
    # for 5 folds
    for f in range(1, 6):
        training_path = "{}/Fold{}/train.txt".format(dataset_fold, f)
        test_path = "{}/Fold{}/test.txt".format(dataset_fold, f)
        train_set = LetorDataset(training_path, FEATURE_SIZE)
        test_set = LetorDataset(test_path, FEATURE_SIZE)


        mp.Process(target=job, args=(click_model, f, train_set, test_set, tau)).start()
