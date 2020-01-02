import sys
sys.path.append('../')
from dataset.LetorDataset import LetorDataset
from ranker.ESLinearRanker import ESLinearRanker
from ranker.NESLinearRanker import NESLinearRanker
from clickModel.SDBN import SDBN
from utils import evl_tool
import numpy as np
import multiprocessing as mp
import pickle
import pdb
import copy

class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


def run(train_set, test_set, ranker, num_interation, click_model, num_rankers):
    ndcg_scores = []
    cndcg_scores = []
    query_set = train_set.get_all_querys()
    index = np.random.randint(query_set.shape[0], size=num_interation)

    batch_size = 100
    iterated = 0
    # for j in range(0, num_interation // batch_size):
    for i in range(num_interation):
        R = np.zeros((num_rankers,))
        canditate_rankers = ranker.sample_new_pop(num_rankers)
        canditate_rankers[0] = ranker.get_current_weights()

        new_rankers = []
        for weights in canditate_rankers:
            new_ranker = copy.deepcopy(ranker)
            new_ranker.assign_weights(weights)
            new_rankers.append(new_ranker)
        # for i in index[j * batch_size:j * batch_size + batch_size]:
        # record = []
        iterated += 1
        qid = query_set[index[i]]

        # result_list = ranker.get_query_result_list(train_set, qid)
        query_features = train_set.get_all_features_by_query(qid)
        (result_list, a) = ranker.probabilistic_multileave(new_rankers, query_features, 10)


        clicked_doc, click_label, _ = click_model.simulate(qid, result_list, train_set)


        cndcg = evl_tool.query_ndcg_at_k(train_set, result_list, qid, 10)
        cndcg_scores.append(cndcg)

        # if no clicks, skip.
        if len(clicked_doc) == 0:
            # all_result = ranker.get_all_query_result_list(test_set)
            # ndcg = evl_tool.average_ndcg_at_k(test_set, all_result, 10)
            # cndcg = evl_tool.query_ndcg_at_k(train_set, result_list, qid, 10)
            #
            # ndcg_scores.append(ndcg)
            # cndcg_scores.append(cndcg)
            continue

        # flip click label. exp: [1,0,1,0,0] -> [0,1,0,0,0]
        # last_click = np.where(click_label == 1)[0][-1]
        # click_label[:last_click + 1] = 1 - click_label[:last_click + 1]

        # bandit record
        # record.append((qid, result_list, click_label, ranker.get_current_weights()))
        # snips = ranker.get_SNIPS(canditate_rankers, record, train_set)

        R = np.array(ranker.probabilistic_multileave_outcome(result_list, new_rankers, click_label, query_features))

        # if snips is not None:
        #     R += snips

        #R = R[1:]
        R = np.interp(R, (R.min(), R.max()), (0, 1))

        dmu = np.zeros((FEATURE_SIZE))
        dcov = np.zeros((FEATURE_SIZE, FEATURE_SIZE))
        Fmu = np.zeros((FEATURE_SIZE, FEATURE_SIZE))
        Fcov = np.zeros((FEATURE_SIZE, FEATURE_SIZE))
        for k in range(0, num_rankers):
            difference = canditate_rankers[k] - ranker.mu
            difference = difference.reshape((FEATURE_SIZE,1))

            covinv = np.linalg.inv(ranker.cov)
            dlogPidmu = np.matmul(covinv, difference)
            Fmu += np.matmul(dlogPidmu, dlogPidmu.T)
            dlogPidmu = dlogPidmu.flatten()
            dlogPidcov = -0.5 * covinv + 0.5 * np.matmul(np.matmul(np.matmul(covinv, difference), difference.T), covinv)
            Fcov += np.matmul(dlogPidcov, dlogPidcov.T)
            dmu += dlogPidmu * R[k]
            dcov += dlogPidcov * R[k]
        dmu /= num_rankers
        dcov /= num_rankers
        Fmu /= num_rankers
        Fcov /= num_rankers
        mu_update = np.matmul(np.linalg.inv(Fmu), dmu)
        cov_update = np.matmul(np.linalg.inv(Fcov), dcov)
        ranker.update(mu_update, cov_update)
        #ranker.assign_weights(canditate_rankers[np.argmax(R)])
        ranker.assign_weights(ranker.mu)
        # print(ranker.mu)
        all_result = ranker.get_all_query_result_list(test_set)
        ndcg = evl_tool.average_ndcg_at_k(test_set, all_result, 10)


        #ndcg = [ndcg] * batch_size
        #ndcg_scores.extend(ndcg)
        # print(len(ndcg_scores))
        final_weight = ranker.get_current_weights()
        print(iterated, ndcg, cndcg)

    return ndcg_scores, cndcg_scores, final_weight


def job(model_type, f, train_set, test_set, tau, sigma, gamma, num_rankers, learning_rate_decay, output_fold):
    if model_type == "perfect":
        pc = [0.0, 0.5, 1.0]
        ps = [0.0, 0.0, 0.0]
    elif model_type == "navigational":
        pc = [0.05, 0.5, 0.95]
        ps = [0.2, 0.5, 0.9]
    elif model_type == "informational":
        pc = [0.4, 0.7, 0.9]
        ps = [0.1, 0.3, 0.5]

    """if model_type == "perfect":
        pc = [0.0, 0.2, 0.4, 0.8, 1.0]
        ps = [0.0, 0.0, 0.0, 0.0, 0.0]
    elif model_type == "navigational":
        pc = [0.05, 0.3, 0.5, 0.7, 0.95]
        ps = [0.2, 0.3, 0.5, 0.7, 0.9]
    elif model_type == "informational":
        pc = [0.4, 0.6, 0.7, 0.8, 0.9]
        ps = [0.1, 0.2, 0.3, 0.4, 0.5]"""

    cm = SDBN(pc, ps)

    for r in range(1, 26):
        # np.random.seed(r)
        mu = np.zeros((FEATURE_SIZE))
        cov = np.eye(FEATURE_SIZE) * 0.1

        ranker = NESLinearRanker(FEATURE_SIZE, Learning_rate, mu, cov, sigma, tau, gamma, learning_rate_decay=learning_rate_decay)
        print("ES fold{} {} run{} start!".format(f, model_type, r))
        ndcg_scores, cndcg_scores, final_weight = run(train_set, test_set, ranker, NUM_INTERACTION, cm, num_rankers)
        with open(
                "{}/fold{}/{}_sigma{}_run{}_ndcg.txt".format(output_fold, f, model_type, sigma, r),
                "wb") as fp:
            pickle.dump(ndcg_scores, fp)
        with open(
                "{}/fold{}/{}_sigma{}_run{}_cndcg.txt".format(output_fold, f, model_type, sigma, r),
                "wb") as fp:
            pickle.dump(cndcg_scores, fp)
        with open(
                "{}/fold{}/{}_sigma{}_run{}_final_weight.txt".format(output_fold, f, model_type, sigma, r),
                "wb") as fp:
            pickle.dump(final_weight, fp)
        print("ES sigma{} fold{} {} run{} finished!".format(output_fold, sigma, f, model_type, r))


if __name__ == "__main__":

    FEATURE_SIZE = 46
    NUM_INTERACTION = 10000
    # click_models = ["informational", "navigational", "perfect"]
    click_models = ["informational"]
    Learning_rate = 1

    #dataset_fold = "../datasets/MSLR-WEB10K"
    #output_fold = "../results/ES/MSLR10K"
    dataset_fold = "../datasets/2007_mq_dataset"
    output_fold = "mq2007"

    num_rankers = 50
    tau = 1
    gamma = 1
    learning_rate_decay = 0.995
    sigma = 0.1

    # for 5 folds
    for f in range(1, 2):
        training_path = "{}/Fold{}/train.txt".format(dataset_fold, f)
        test_path = "{}/Fold{}/test.txt".format(dataset_fold, f)
        train_set = LetorDataset(training_path, FEATURE_SIZE, query_level_norm=True)
        test_set = LetorDataset(test_path, FEATURE_SIZE, query_level_norm=True)

        # for 3 click_models
        for click_model in click_models:
            mp.Process(target=job, args=(click_model, f, train_set, test_set,
                                         tau, sigma, gamma, num_rankers, learning_rate_decay,
                                         output_fold)).start()
