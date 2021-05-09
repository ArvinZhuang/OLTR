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
    knots = np.linspace(x[0], x[-1], 1000)
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
    plot_result = np.cumsum(result, axis=1)

    n = result.shape[1]
    result_mean = np.mean(plot_result, axis=1)
    result_std_err = sem(plot_result, axis=1)
    result_h = result_std_err * t.ppf((1 + 0.95) / 2, n - 1)
    result_low = np.subtract(result_mean, result_h)
    result_high = np.add(result_mean, result_h)

    plt.plot(range(num_interactions), smooth(result_mean, num_interactions), color=COLORS[color], alpha=1)

    plt.fill_between(range(num_interactions), smooth(result_low, num_interactions), smooth(result_high, num_interactions), color=COLORS[color], alpha=0.2)
    color_index += 1
    cndcgs = []
    for start, end in intervals:
        cndcg = 0
        for i in range(start, end):
            cndcg += 0.9995 ** i * result[i]
        cndcgs.append(cndcg / (end - start))
    print(parameter, np.mean(cndcg))

    plt.figure(1)
    return cndcg


if __name__ == "__main__":
    ############## plot different reward function ####
    # path1 = "results/mslr10k/MDP_with_SGD_optimizer/MDP_001_both"
    # path2 = "results/mslr10k/MDP_with_SGD_optimizer/MDP_001_both_naive"
    # path3 = "results/mslr10k/MDP_with_SGD_optimizer/MDP_001_positive"
    # path4 = "results/mslr10k/MDP_with_SGD_optimizer/MDP_001_positive_naive"
    # path5 = "results/mslr10k/MDP_with_SGD_optimizer/MDP_001_negative"
    # path6 = "results/mslr10k/MDP_with_SGD_optimizer/MDP_001_negative_naive"

    ############## plot different propensities ####
    # path1 = "results/mslr10k/MDP_with_SGD_optimizer/MDP_001_both_naive"
    # path2 = "results/mslr10k/MDP_with_SGD_optimizer/MDP_001_both_propensity0.5"
    # path3 = "results/mslr10k/MDP_with_SGD_optimizer/MDP_001_both"
    # path4 = "results/mslr10k/MDP_with_SGD_optimizer/MDP_001_both_propensity1.5"
    # path5 = "results/mslr10k/MDP_with_SGD_optimizer/MDP_001_both_propensity2.0"


    ############## plot different algorithms ####
    path1 = "results/mslr10k/MDP_with_SGD_optimizer/MDP_001_both"
    path2 = "results/mslr10k/long_term_200k/PDGD"
    path3 = "results/mslr10k/DBGD"
    path4 = "results/mslr10k/PMGD"
    path5 = "results/mslr10k/COLTR"
    path7 = "results/mslr10k/MDP_with_SGD_optimizer/MDP_001_both_one_at_time"

    # path1 = "results/yahoo/MDP_with_SGD_optimizer/MDP_0002_both"
    # path2 = "results/yahoo/PDGD"
    # path3 = "results/yahoo/DBGD"
    # path4 = "results/yahoo/PMGD"
    # path5 = "results/yahoo/COLTR"

    # path1 = "results/istella/MDP_with_SGD_optimizer/MDP_001_both"
    # path2 = "results/istella/PDGD"
    # path3 = "results/istella/DBGD"
    # path4 = "results/istella/PMGD"
    # path5 = "results/istella/COLTR"

    # path1 = "results/mslr10k/MDP_with_SGD_optimizer/MDP_001_positive_naive_gamma0_variance"
    # path2 = "results/mslr10k/MDP_with_SGD_optimizer/MDP_001_positive_naive_gamma1_variance"

    folds = list(range(1, 6))
    runs = list(range(1, 11))
    # intervals = [(0, 10000), (10000, 100000)]
    intervals = [(0, 100000)]
    click_model = 'perfect'

    # parameters = ["ReOLTR", "PDGD", "PDGD", "MDP_pos+neg"]
    parameters = ["ReOLTR", "ReOLTR_new"]
    num_interactions = 100000
    print(click_model)
    # l1 = plot(path1, "ReOLTR", folds, runs, click_model, num_interactions, 1, intervals)
    # l2 = plot(path2, "PDGD", folds, runs, click_model, num_interactions, 2, intervals)
    # plot(path3, "DBGD", folds, runs, click_model, num_interactions, 3, intervals)
    # plot(path4, "PMGD", folds, runs, click_model, num_interactions, 4, intervals)
    # plot(path5, "COLTR", folds, runs, click_model, num_interactions, 5, intervals)
    # plot(path7, "ROLTR_new", folds, runs, click_model, num_interactions, 5, intervals)
    # print(ttest(l1[0], l2[0]), ttest(l1[1], l2[1]))
    # print(ttest(l1, l2))
    # plt.ylabel('NDCG')
    # plt.xlabel('EPOCH')

    click_model = 'informational'

    # parameters = ["ReOLTR", "PDGD", "PDGD", "MDP_pos+neg"]
    # parameters = ["ReOLTR", "ReOLTR_new"]
    # num_interactions = 100000
    # print(click_model)
    l1 = plot(path1, "ReOLTR", folds, runs, click_model, num_interactions, 1, intervals)
    # l2 = plot(path2, "PDGD", folds, runs, click_model, num_interactions, 2, intervals)
    # plot(path3, "DBGD", folds, runs, click_model, num_interactions, 3, intervals)
    # plot(path4, "PMGD", folds, runs, click_model, num_interactions, 4, intervals)
    # plot(path5, "COLTR", folds, runs, click_model, num_interactions, 5, intervals)
    plot(path7, "ROLTR_new", folds, runs, click_model, num_interactions, 5, intervals)

    plt.legend(parameters, loc='lower right')
    # plt.show()
