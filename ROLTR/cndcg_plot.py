import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, t

COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

def plot(path, parameter, folds, runs, click_model, num_interactions, color, intervals):
    color_index = 0
    result = np.zeros(num_interactions)
    for f in folds:
        for r in runs:
            with open("{}/fold{}/{}_run{}_cndcg.txt".format(path, f, click_model, r),
                      "rb") as fp:
                data = pickle.load(fp)
                data = np.array(data[:num_interactions])
                result = np.vstack((result, data))
    result = result[1:].T
    result_mean = np.mean(result, axis=1)
    result_std_err = sem(result, axis=1)
    result_h = result_std_err * t.ppf((1 + 0.95) / 2, 25 - 1)
    result_low = np.subtract(result_mean, result_h)
    result_high = np.add(result_mean, result_h)

    plt.plot(range(num_interactions), result_mean, color=COLORS[color], alpha=1)
    # plt.fill_between(range(num_interactions), result_low, result_high, color='black', alpha=0.2)
    color_index += 1
    cndcgs = []
    for start, end in intervals:
        cndcg = 0
        for i in range(start, end):
            cndcg += 1 ** i * result_mean[i - 1]
        cndcgs.append(cndcg / (end - start))
    print(parameter, cndcgs)

    plt.figure(1)



if __name__ == "__main__":
    path1 = "results/mslr10k/PDGD"
    # path2 = "results/mslr10k/MDP_001_positive"
    # path3 = "results/mslr10k/MDP_001_negative"
    path4 = "results/mslr10k/MDP_001_both"
    # path5 = "results/mslr10k/MDP_001_both_naive"
    # path6 = "results/mslr10k/MDP_001_negative_naive"
    # path7 = "results/mslr10k/MDP_001_positive_naive"
    # path8 = "results/mslr10k/MDP_001_both_propensity0.5"
    # path9 = "results/mslr10k/MDP_001_both_propensity1.5"
    # path10 = "results/mslr10k//MDP_001_both_propensity2.0"
    # path11 = "results/mslr10k/MDP_listwise_reward/MDP_001_both_gamma0.5"
    # path12 = "results/mslr10k/MDP_listwise_reward/MDP_001_positive_gamma0.5"
    # path13 = "results/mslr10k/MDP_001_both_correct"

    path1 = "results/mq2007/PDGD"
    # path2 = "results/mq2007/MDP_001_positive"
    # path3 = "results/mq2007/MDP_001_negative"
    path4 = "results/mq2007/MDP_001_both"
    # path13 = "results/mq2007/MDP_01_both_correct"
    # path14 = "results/mq2007/MDP_0001_both_correct"
    folds = list(range(1, 6))
    runs = list(range(1, 16))
    intervals = [(0, 10000), (10000, 100000)]
    click_model = 'perfect'

    parameters = ["PDGD", "MDP_positiveDCG", "MDP_negativeDCG", "MDP_pos+neg"]
    num_interactions = 100000

    plot(path1, "PDGD", folds, runs, click_model, num_interactions, 1, intervals)
    # plot(path2, "MDP_positiveDCG", folds, runs, click_model, num_interactions, 2, intervals)
    # plot(path3, "MDP_negativeDCG", folds, runs, click_model, num_interactions, 3, intervals)
    plot(path4, "MDP_pos+neg", folds, runs, click_model, num_interactions, 4, intervals)
    # plot(path5, "MDP_pos+neg_naive", folds, runs, click_model, num_interactions, 4, intervals)
    # plot(path13, "MDP_pos+neg_naive", folds, runs, click_model, num_interactions, 4, intervals)
    # plt.ylabel('NDCG')
    # plt.xlabel('EPOCH')
    # plt.legend(parameters, loc='lower right')
    # plt.show()
