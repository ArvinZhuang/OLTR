import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, t

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
    result = result[1:].T
    result_mean = np.mean(result, axis=1)
    result_std_err = sem(result, axis=1)
    result_h = result_std_err * t.ppf((1 + 0.95) / 2, 25 - 1)
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




if __name__ == "__main__":
    # path1 = "results/mslr10k/PDGD"
    #
    # path2 = "results/mslr10k/MDP_001_positive"
    # path3 = "results/mslr10k/MDP_001_negative"
    # path4 = "results/mslr10k/MDP_001_both"
    # path5 = "results/mslr10k/MDP_001_both_naive"
    # path6 = "results/mslr10k/MDP_001_negative_naive"
    # path7 = "results/mslr10k/MDP_001_positive_naive"
    # path8 = "results/mslr10k/MDP_001_both_propensity0.5"
    # path9 = "results/mslr10k/MDP_001_both_propensity1.5"
    # path10 = "results/mslr10k//MDP_001_both_propensity2.0"
    # path11 = "results/mslr10k/MDP_listwise_reward/MDP_001_both_gamma0.2"
    # path12 = "results/mslr10k/MDP_listwise_reward/MDP_001_positive_gamma0.5"
    # path13 = "results/mslr10k/MDP_001_both_correct"
    path14 = "results/mslr10k/COLTR"
    # path15 = "results/mslr10k/MDP_listwise_reward/MDP_001_positive_gamma1"
    # path16 = "results/mslr10k/test/MDP_001_both_policy_trust"
    # path17 = "results/mslr10k/AC_0001_both"
    path18 = "results/mslr10k/long_term_200k/PDGD"
    path19 = "results/mslr10k/MDP_001_both_Adam"
    path20 = "results/mslr10k/long_term_200k/MDP_001decay_both_Adam"
    # path21 = "results/mslr10k/long_term_200k/MDP_0001_both_Adam"
    # path21 = "results/mslr10k/AC_001_both_noise_critic"
    path22 = "results/mslr10k/unbiased_COLTR"

    path1 = "results/mslr10k/long_term_200k/PDGD"
    path2 = "results/mslr10k/long_term_200k/MDP_001decay_both_Adam"
    path3 = "results/mslr10k/long_term_200k/MDP_001_Adam_positive_naive_gamma0"
    path4 = "results/mslr10k/long_term_200k/MDP_001_Adam_positive_naive_gamma1"
    path5 = "results/mslr10k/long_term_200k/MDP_001_Adam_positive_unbiased_gamma0"
    path6 = "results/mslr10k/long_term_200k/MDP_001_Adam_positive_unbiased_gamma1"
    #
    # path1 = "results/mq2007/PDGD"
    # path2 = "results/mq2007/MDP_001_positive"
    # path3 = "results/mq2007/MDP_001_negative"
    # path4 = "results/mq2007/MDP_001_both"
    # path13 = "results/mq2007/MDP_01_both_correct"
    # path14 = "results/mq2007/MDP_0001_both_correct"

    # path1 = "results/yahoo/PDGD"
    # path2 = "results/yahoo/MDP_0001_Adam_both_gamma0"
    # path3 = "results/yahoo/MDP_0001_Adam_positive_naive_gamma0"
    # path4 = "results/yahoo/MDP_0001_Adam_positive_naive_gamma1"
    # path5 = "results/yahoo/MDP_0001_Adam_positive_gamma0"
    # path6 = "results/yahoo/MDP_0001_Adam_positive_gamma1"
    folds = list(range(1, 6))
    runs = list(range(1, 16))
    click_models = ['informational', "perfect"]
    # click_models = ["perfect"]

    # parameters1 = ["PDGD", "MDP_DCG_unbiased", "MDP_negativeDCG", "MDP_DCG+negativeDCG_unbiased", "MDP_pos+neg_naive",
    #               "MDP_negativeDCG_naive",  "MDP_DCG_naive"]
    parameters1 = ["PDGD", "MDP_DCG+negativeDCG_unbiased_gamma0", "MDP_DCG_naive_gamma0", "MDP_DCG_naive_gamma1",
                   "MDP_DCG_unbiased_gamma0", "MDP_DCG_unbiased_gamma1"]

    parameters2 = ["propensity0.0(naive)", "propensity0.5", "propensity1(true)", "propensity1.5", "propensity2.0" ]
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

            # plot(path14, folds, runs, click_model, num_interaction, 1, plot_index)
            plot(path1, folds, runs, click_model, num_interaction, 7, plot_index)
            plot(path2, folds, runs, click_model, num_interaction, 3, plot_index)
            plot(path3, folds, runs, click_model, num_interaction, 4, plot_index)
            plot(path4, folds, runs, click_model, num_interaction, 2, plot_index)
            plot(path5, folds, runs, click_model, num_interaction, 5, plot_index)
            plot(path6, folds, runs, click_model, num_interaction, 0, plot_index)

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
