import os
import sys

sys.path.append('../')
from dataset import LetorDataset
import numpy as np
from clickModel.PBM import PBM
from ranker.MDPRankerV2 import MDPRankerV2
from utils import evl_tool
from utils.utility import get_DCG_rewards, get_DCG_MDPrewards
import multiprocessing as mp
import pickle


# %%
def run(train_set, test_set, ranker, eta, gamma, reward_method, num_interation, click_model):
    vars = []
    query_set = train_set.get_all_querys()
    index = np.random.randint(query_set.shape[0], size=num_interation)

    for i in index:
        qid = query_set[i]
        result_list = ranker.get_query_result_list(train_set, qid)
        clicked_doces, click_labels, _ = click_model.simulate(qid, result_list, train_set)

        # if no click data, skip this session
        if len(clicked_doces) == 0:
            vars.append(0)
            continue

        propensities = np.power(np.divide(1, np.arange(1.0, len(click_labels) + 1)), eta)

        rewards = get_DCG_MDPrewards(click_labels, propensities, reward_method, gamma=gamma)

        gradient_var = ranker.update_policy(qid, result_list, rewards, train_set)
        vars.append(gradient_var)

    return vars


def job(model_type, learning_rate, eta, gamma, reward_method, f, train_set, test_set, num_features, output_fold):

    if model_type == "perfect":
        pc = [0.0, 0.2, 0.4, 0.8, 1.0]
        ps = [0.0, 0.0, 0.0, 0.0, 0.0]
    elif model_type == "navigational":
        pc = [0.05, 0.3, 0.5, 0.7, 0.95]
        ps = [0.2, 0.3, 0.5, 0.7, 0.9]
    elif model_type == "informational":
        pc = [0.4, 0.6, 0.7, 0.8, 0.9]
        ps = [0.1, 0.2, 0.3, 0.4, 0.5]
    # #
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

    for r in range(1, 16):
        # np.random.seed(r)
        ranker = MDPRankerV2(256, num_features, learning_rate)
        print("MDP Adam MSLR10K fold{} {} eta{} reward {} run{} start!".format(f, model_type, eta, reward_method, r))
        vars = run(train_set, test_set, ranker, eta, gamma, reward_method, NUM_INTERACTION, cm)
        os.makedirs(os.path.dirname("{}/fold{}/".format(output_fold, f)),
                    exist_ok=True)  # create directory if not exist
        with open(
                "{}/fold{}/{}_run{}.txt".format(output_fold, f, model_type, r),
                "wb") as fp:
            pickle.dump(vars, fp)

        print("MDP MSLR10K fold{} {} eta{} reward{} run{} done!".format(f, model_type, eta, reward_method, r))


if __name__ == "__main__":

    FEATURE_SIZE = 136
    NUM_INTERACTION = 100000
    learning_rate = 0.01
    eta = 0
    gamma = 1
    reward_method = "positive"
    click_models = ["informational", "perfect"]
    # click_models = ["informational"]
    # dataset_fold = "../datasets/2007_mq_dataset"
    dataset_fold = "../datasets/MSLR10K"
    output_fold = "results/mslr10k/MDP_with_SGD_optimizer/MDP_001_positive_naive_gamma1_variance"
    print("reward:", reward_method, "lr:", learning_rate, "eta:", eta, output_fold, "gamma", gamma)
    # for 5 folds
    for f in range(1, 6):
        # training_path = "{}/set1.train.txt".format(dataset_fold)
        # test_path = "{}/set1.test.txt".format(dataset_fold)
        training_path = "{}/Fold{}/train.txt".format(dataset_fold, f)
        test_path = "{}/Fold{}/test.txt".format(dataset_fold, f)
        train_set = LetorDataset(training_path, FEATURE_SIZE, query_level_norm=True)
        test_set = LetorDataset(test_path, FEATURE_SIZE, query_level_norm=True)
        # %%
        processors = []
        # for 3 click_models
        for click_model in click_models:
            p = mp.Process(target=job, args=(click_model, learning_rate, eta, gamma, reward_method, f, train_set, test_set, FEATURE_SIZE, output_fold))
            p.start()
            processors.append(p)
    for p in processors:
        p.join()
