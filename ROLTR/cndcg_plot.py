import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, t
from utils.evl_tool import ttest
from scipy.interpolate import make_lsq_spline, BSpline
COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

def smooth(y, num_interactions):
    x = np.linspace(0, num_interactions, num_interactions)
    k = 4
    knots = np.linspace(x[0], x[-1], 9)
    t = np.r_[(x[0],) * k,
              knots,
              (x[-1],) * k]
    spl = make_lsq_spline(x, y, t, k)
    return spl(x)

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
    # result_lists = []
    # for start, end in intervals:
    #     result_lists.append(np.mean(result[1:, start:end], axis=1))

    result = result[1:].T
    n = result.shape[1]
    result_mean = np.mean(result, axis=1)
    result_std_err = sem(result, axis=1)
    result_h = result_std_err * t.ppf((1 + 0.95) / 2, n - 1)
    result_low = np.subtract(result_mean, result_h)
    result_high = np.add(result_mean, result_h)

    plt.plot(range(num_interactions), smooth(result_mean, num_interactions), color=COLORS[color], alpha=1)

    # plt.fill_between(range(num_interactions), smooth(result_low, num_interactions), smooth(result_high, num_interactions), color=COLORS[color], alpha=0.2)
    color_index += 1
    cndcgs = []
    print(result_mean.shape)
    for start, end in intervals:
        cndcg = 0
        for i in range(start, end):
            cndcg += 0.9999 ** i * result[i - 1]
        cndcgs.append(cndcg / (end - start))
    print(parameter, np.mean(cndcg))

    plt.figure(1)
    return cndcg


if __name__ == "__main__":
    path1 = "results/mslr10k/long_term_200k/PDGD"
    path2 = "results/mslr10k/long_term_200k/PDGD_eta2"
    # path2 = "results/mslr10k/long_term_200k/MDP_001decay_both_Adam"
    # path2 = "results/mslr10k/MDP_001_both_Adam"
    # path3 = "results/mslr10k/long_term_200k/MDP_001_Adam_positive_naive_gamma0"
    # path4 = "results/mslr10k/long_term_200k/MDP_001_Adam_positive_naive_gamma1"
    path5 = "results/mslr10k/long_term_200k/MDP_001_Adam_positive_unbiased_gamma0"
    path6 = "results/mslr10k/long_term_200k/MDP_001_Adam_positive_unbiased_gamma1"
    path7 = "results/mslr10k/long_term_200k/MDP_001_Adam_positive_unbiased_gamma01"
    path8 = "results/mslr10k/long_term_200k/MDP_001_Adam_positive_unbiased_gamma05"

    # path1 = "results/mslr10k/long_term_1m/PDGD"
    # path2 = "results/mslr10k/long_term_1m/MDP_0001_Adam_both_unbiased_gamma0"

    # path1 = "results/mq2007/PDGD"
    # path2 = "results/mq2007/MDP_001_positive"
    # path3 = "results/mq2007/MDP_001_negative"
    # path4 = "results/mq2007/MDP_001_both"
    # path13 = "results/mq2007/MDP_01_both_correct"
    # path14 = "results/mq2007/MDP_0001_both_correct"

    # path1 = "results/yahoo/PDGD"
    # path2 = "results/yahoo/MDP_001decay_Adam_both_gamma0"
    # path2 = "results/yahoo/MDP_0001_Adam_both_gamma0"
    # path3 = "results/yahoo/MDP_0001_Adam_positive_gamma0"
    # path4 = "results/yahoo/MDP_001decay_Adam_both_gamma0"

    # path1 = "results/mq2007/PDGD"
    # path2 = "results/mq2007/MDP_001_Adam_both_gamma0"
    path2 = "results/mslr10k/MDP_with_SGD_optimizer/MDP_001_both"
    folds = list(range(1, 6))
    runs = list(range(1, 16))
    # intervals = [(0, 10000), (10000, 100000)]
    intervals = [(0, 100000)]
    click_model = 'informational'

    parameters = ["PDGD", "MDP_positiveDCG", "MDP_negativeDCG", "MDP_pos+neg"]
    num_interactions = 100000

    l1 = plot(path1, "PDGD", folds, runs, click_model, num_interactions, 1, intervals)
    l2 = plot(path2, "MDP_pos+neg_gamma0", folds, runs, click_model, num_interactions, 2, intervals)
    # print(ttest(l1[0], l2[0]), ttest(l1[1], l2[1]))
    print(ttest(l1, l2))
    # plot(path3, "MDP_pos_naive_gamma0", folds, runs, click_model, num_interactions, 3, intervals)
    # plot(path4, "MDP_pos_naive_gamma1", folds, runs, click_model, num_interactions, 4, intervals)
    # plot(path5, "MDP_pos_unbiased_gamma0", folds, runs, click_model, num_interactions, 4, intervals)
    # plot(path6, "MDP_pos_unbiased_gamma1", folds, runs, click_model, num_interactions, 4, intervals)
    # plot(path19, "MDP_pos+neg_naive", folds, runs, click_model, num_interactions, 4, intervals)
    # plt.ylabel('NDCG')
    # plt.xlabel('EPOCH')
    plt.legend(parameters, loc='lower right')
    # plt.show()
