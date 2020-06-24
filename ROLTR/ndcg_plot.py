import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, t
from utils.evl_tool import ttest

COLORS = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'k']


def plot(path, folds, runs, click_model, num_interactions, color, plot_ind):
    print("click model:", click_model)
    subplot = plt.subplot(2, 2, plot_ind+1)
    plt.title(click_model, loc='left', position=(0.03, 0.9))


    result = np.zeros(int(num_interactions/1000))
    for f in folds:
        for r in runs:
            with open("{}/fold{}/{}_run{}_ndcg.txt".format(path, f, click_model, r),
                      "rb") as fp:
                data = pickle.load(fp)
                data = np.array(data[:int(num_interactions/1000)])
                result = np.vstack((result, data))
    result_list = result[1:, -1]
    result = result[1:].T
    n = result.shape[1]
    result_mean = np.mean(result, axis=1)
    result_std_err = sem(result, axis=1)
    result_h = result_std_err * t.ppf((1 + 0.95) / 2, n - 1)
    result_low = np.subtract(result_mean, result_h)
    result_high = np.add(result_mean, result_h)

    plt.plot(range(0, num_interactions, 1000), result_mean, color=COLORS[color], alpha=1)

    plt.fill_between(range(0, num_interactions, 1000), result_low, result_high, color=COLORS[color], alpha=0.2)

    if plot_ind % 2 == 0 :
        plt.ylabel('NDCG')
    if plot_ind // 2 == 1 :
        plt.xlabel('impressions')
    # plt.ylim([0.0, 0.45])
    # plt.legend(parameters, loc='lower right')
    print("result path:", path, result_mean[-1])
    return result_list



if __name__ == "__main__":
    path1 = "results/mslr10k/long_term_200k/PDGD"
    # path2 = "results/mslr10k/COLTR"
    path2 = "results/mslr10k/long_term_200k/PDGD_eta2"
    # path2 = "results/mslr10k/MDP_001_both_Adam"
    # path2 = "results/mslr10k/long_term_200k/MDP_0001_both_Adam"
    # path1 = "results/mslr10k/long_term_1m/PDGD"
    # path2 = "results/mslr10k/long_term_1m/MDP_0001_Adam_both_unbiased_gamma0"
    # path3 = "results/mslr10k/long_term_200k/MDP_001decay_both_Adam"
    path3 = "results/mslr10k/long_term_200k/MDP_001_Adam_positive_naive_gamma0"
    path4 = "results/mslr10k/long_term_200k/MDP_001_Adam_positive_naive_gamma1"
    path5 = "results/mslr10k/long_term_200k/MDP_001_Adam_positive_unbiased_gamma0"
    path6 = "results/mslr10k/long_term_200k/MDP_001_Adam_positive_unbiased_gamma1"
    path7 = "results/mslr10k/long_term_200k/MDP_001_Adam_positive_unbiased_gamma01"
    # path8 = "results/mslr10k/long_term_200k/MDP_001_Adam_positive_unbiased_gamma05"

    path1 = "results/yahoo/PDGD"
    path2 = "results/yahoo/MDP_with_SGD_optimiser/MDP_001_both_unbiased"
    # path2 = "results/yahoo/MDP_0001_Adam_both_gamma0"
    # path2 = "results/yahoo/MDP_001decay_Adam_both_gamma0"
    # path2 = "results/yahoo/MDP_001_Adam_both_gamma0"
    # path3 = "results/yahoo/MDP_0001_Adam_positive_naive_gamma0"
    # path4 = "results/yahoo/MDP_0001_Adam_positive_naive_gamma1"
    # path5 = "results/yahoo/MDP_0001_Adam_positive_gamma0"
    # path6 = "results/yahoo/MDP_0001_Adam_positive_gamma1"

    # path1 = "results/mq2007/PDGD"
    # path2 = "results/mq2007/MDP_001_Adam_both_gamma0"

    # path1 = "results/mslr10k/long_term_200k/PDGD"
    # path2 = "results/mslr10k/long_term_200k/PDGD_eta0"
    # path3 = "results/mslr10k/long_term_200k/PDGD_eta2"
    # path2 = "results/mslr10k/long_term_200k/MDP_001_Adam_both_unbiased_gamma0_eta2_clipped"
    # path3 = "results/mslr10k/MDP_001_both_Adam"
    # path4 = "results/mslr10k/long_term_200k/PDGD"

    # path2 = "results/mslr10k/MDP_with_SGD_optimizer/MDP_001_both"


    folds = list(range(1, 2))
    runs = list(range(1, 2))
    click_models = ['informational', "perfect"]


    parameters1 = ["PDGD", "MDP_DCG+negativeDCG_unbiased_gamma0",
                   "MDP_DCG_unbiased_gamma0", "MDP_DCG_unbiased_gamma1", "MDP_DCG_unbiased_gamma01"]
    # parameters1 = ["PDGD", "MDP_DCG_unbiased_gamma0", "MDP_DCG_unbiased_gamma1", "MDP_DCG_unbiased_gamma01", "MDP_DCG_unbiased_gamma05"]
    parameters2 = ["propensity0.0(naive)", "propensity0.5", "propensity1(true)", "propensity1.5", "propensity2.0"]
    num_interactions = [10000, 100000]


    # plot different rewards
    f = plt.figure(1, figsize=(12, 8))

    plot_index = 0
    for click_model in click_models:
        for num_interaction in num_interactions:
            # plot(path1, folds, runs, click_model, num_interaction, 7, plot_index)
            # # plot(path7, folds, runs, click_model, num_interaction, 6, plot_index)
            # plot(path4, folds, runs, click_model, num_interaction, 0, plot_index)
            # plot(path2, folds, runs, click_model, num_interaction, 3, plot_index)
            # plot(path3, folds, runs, click_model, num_interaction, 1, plot_index)
            # plot(path4, folds, runs, click_model, num_interaction, 2, plot_index)
            # plot(path5, folds, runs, click_model, num_interaction, 5, plot_index)
            # plot(path6, folds, runs, click_model, num_interaction, 0, plot_index)
            # plot(path7, folds, runs, click_model, num_interaction, 6, plot_index)

            # plot(path2, folds, runs, click_model, num_interaction, 5, plot_index)
            l1 = plot(path1, folds, runs, click_model, num_interaction, 7, plot_index)
            l2 = plot(path2, folds, runs, click_model, num_interaction, 2, plot_index)
            # l3 = plot(path3, folds, runs, click_model, num_interaction, 3, plot_index)
            # l3 = plot(path6, folds, runs, click_model, num_interaction, 4, plot_index)
            # plot(path7, folds, runs, click_model, num_interaction, 1, plot_index)
            # plot(path5, folds, runs, click_model, num_interaction, 5, plot_index)
            # plot(path6, folds, runs, click_model, num_interaction, 0, plot_index)
            # plot(path7, folds, runs, click_model, num_interaction, 2, plot_index)
            # plot(path8, folds, runs, click_model, num_interaction, 4, plot_index)
            # print(ttest(l1, l2))
            plot_index += 1
            print()
    plt.legend(parameters1, loc='lower right')

    # f.subplots_adjust(wspace=0.3, hspace=0.3)
    # plt.savefig('DCG_unbiased.png', bbox_inches='tight')
    #
    # plt.figure(2)
    # # plot different propensities
    # plot_index = 0
    # for click_model in click_models:
    #     for num_interaction in num_interactions:
    #         plot(path5, folds, runs, click_model, num_interaction, 2, plot_index)
    #         plot(path8, folds, runs, click_model, num_interaction, 1, plot_index)
    #         plot(path4, folds, runs, click_model, num_interaction, 0, plot_index)
    #         plot(path9, folds, runs, click_model, num_interaction, 3, plot_index)
    #         plot(path10, folds, runs, click_model, num_interaction, 4, plot_index)
    #
    #         plot_index += 1
    #         print()
    # plt.legend(parameters2, loc='lower right')
    plt.show()
