import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, t
from utils.evl_tool import ttest

COLORS = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'k']


def plot(path, folds, runs, click_model, num_interactions, color, plot_ind, linestyle, marker):
    print("click model:", click_model)
    plt.subplot(1, 2, plot_ind + 1)
    plt.title(click_model, loc='left', position=(0.03, 0.9))

    result = np.zeros(int(num_interactions / 1000))
    for f in folds:
        for r in runs:
            with open("{}/fold{}/{}_run{}_ndcg.txt".format(path, f, click_model, r),
                      "rb") as fp:
                data = pickle.load(fp)
                data = np.array(data[:int(num_interactions / 1000)])
                result = np.vstack((result, data))
    result_list = result[1:, -1]
    result = result[1:].T
    n = result.shape[1]
    result_mean = np.mean(result, axis=1)
    result_std_err = sem(result, axis=1)
    result_h = result_std_err * t.ppf((1 + 0.95) / 2, n - 1)
    result_low = np.subtract(result_mean, result_h)
    result_high = np.add(result_mean, result_h)

    plt.plot(range(0, num_interactions, 1000), result_mean, color=COLORS[color], alpha=1,
             linestyle=linestyle, marker=marker, markevery=10, markersize=10)

    plt.fill_between(range(0, num_interactions, 1000), result_low, result_high, color=COLORS[color], alpha=0.2)

    if plot_ind % 2 == 0:
        plt.ylabel('NDCG')
    if plot_ind // 2 == 1:
        plt.xlabel('impressions')
    # plt.ylim([0.2, 0.45])
    # plt.yticks(np.arange(0.55, 0.8, 0.05))
    plt.ylim([0.3, 0.55])
    print("result path:", path, result_mean[-1])
    return result_list


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
    # path1 = "results/mslr10k/MDP_with_SGD_optimizer/MDP_001_both"
    # path2 = "results/mslr10k/MDP_with_SGD_optimizer/MDP_0001_both_unbiased_pairwise"
    # path3 = "results/mslr10k/long_term_200k/PDGD"
    # path4 = "results/mslr10k/COLTR"
    # path5 = "results/mslr10k/PMGD"

    # path1 = "results/yahoo/MDP_with_SGD_optimizer/MDP_0001_both_unbiased"
    # path2 = "results/yahoo/MDP_with_SGD_optimizer/MDP_0001_both_unbiased_pairwise"
    # path3 = "results/yahoo/PDGD"
    # path4 = "results/yahoo/COLTR_gamma1"
    # path5 = "results/yahoo/PMGD"

    path1 = "results/mq2007/MDP_001_both"
    path2 = "results/mq2007/MDP_0001_both_pairwise"
    path3 = "results/mq2007/PDGD"
    path4 = "results/mq2007/COLTR_gamma1"
    path5 = "results/mq2007/PMGD"

    folds = list(range(1, 6))
    runs = list(range(1, 3))
    click_models = ["informational", "perfect"]

    ############## plot different reward function ####
    # legends = ["$R_{IPS^{+-}}$",
    #            "$R_{NAIVE^{+-}}$",
    #            "$R_{IPS^{+}}$",
    #            "$R_{NAIVE^{+}}$",
    #            "$R_{IPS^{-}}$",
    #            "$R_{NAIVE^{-}}$"]

    ############## plot different algorithms ####
    # legends = ["$\eta=0$ (naive)",
    #            "$\eta=0.5$",
    #            "$\eta=1.0$ (true)",
    #            "$\eta=1.5$",
    #            "$\eta=2.0$"]

    ############## plot different algorithms ####
    legends = ["ReOLTR",
               "ReOLTR_pairwise",
               "PDGD",
               "COLTR",
               "PMGD"]


    num_interactions = [100000]

    # plot different rewards
    f = plt.figure(1, figsize=(12, 4))

    plot_index = 0
    for click_model in click_models:
        for num_interaction in num_interactions:
            ############## plot different reward function ####
            # plot(path1, folds, runs, click_model, num_interaction, 0, plot_index, '-', None)
            # plot(path2, folds, runs, click_model, num_interaction, 0, plot_index, '--', None)
            # plot(path3, folds, runs, click_model, num_interaction, 5, plot_index, '-', None)
            # plot(path4, folds, runs, click_model, num_interaction, 5, plot_index, '--', None)
            # plot(path5, folds, runs, click_model, num_interaction, 4, plot_index, '-', None)
            # plot(path6, folds, runs, click_model, num_interaction, 4, plot_index, '--', None)

            ############## plot different algorithms ####
            # plot(path1, folds, runs, click_model, num_interaction, 0, plot_index, '--', None)
            # plot(path2, folds, runs, click_model, num_interaction, 0, plot_index, '--', '+')
            # plot(path3, folds, runs, click_model, num_interaction, 0, plot_index, '-', None)
            # plot(path4, folds, runs, click_model, num_interaction, 0, plot_index, '--', 'x')
            # plot(path5, folds, runs, click_model, num_interaction, 0, plot_index, '--', '1')

            ############## plot different algorithms ####
            l1 = plot(path1, folds, runs, click_model, num_interaction, 0, plot_index, '-', None)
            plot(path2, folds, runs, click_model, num_interaction, 3, plot_index, '-', None)
            l2 = plot(path3, folds, runs, click_model, num_interaction, 2, plot_index, '-', None)
            # plot(path4, folds, runs, click_model, num_interaction, 6, plot_index, '-', None)
            # plot(path5, folds, list(range(1, 2)), click_model, num_interaction, 1, plot_index, '-', None)

            # print(ttest(l1, l2))
            plot_index += 1
            print()
    plt.legend(legends, loc='lower right')

    f.subplots_adjust(wspace=0.2, hspace=0.3)
    plt.savefig('mq2007.png', bbox_inches='tight')

    plt.show()
