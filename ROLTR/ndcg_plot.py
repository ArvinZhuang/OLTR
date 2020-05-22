import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, t

COLORS = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'k']


def plot(path, folds, runs, click_model, num_interactions, color, plot_ind):
    print("click model:", click_model)
    plt.subplot(1, 2, plot_ind+1)
    plt.title(click_model)

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

    # plt.fill_between(range(0, num_interactions, 1000), result_low, result_high, color=COLORS[color], alpha=0.2)
    plt.ylabel('NDCG')
    plt.xlabel('impressions')
    # plt.ylim([0.2, 0.45])
    # plt.legend(parameters, loc='lower right')
    print("result path:", path, result_mean[-1])




if __name__ == "__main__":
    path1 = "results/mslr10k/PDGD"
    path2 = "results/mslr10k/MDP_001_positive"
    path3 = "results/mslr10k/MDP_001_negative"
    path4 = "results/mslr10k/MDP_001_both"
    path5 = "results/mslr10k/MDP_001_both_naive"
    path6 = "results/mslr10k/MDP_001_negative_naive"
    path7 = "results/mslr10k/MDP_001_positive_naive"
    path8 = "results/mslr10k/MDP_001_both_propensity1.5"


    # path1 = "results/mq2007/PDGD"
    # path2 = "results/mq2007/MDP_001_positive"
    # path3 = "results/mq2007/MDP_001_negative"
    # path4 = "results/mq2007/MDP_001_both"

    folds = list(range(1, 6))
    runs = list(range(1, 3))
    click_models = ['informational', "perfect"]

    parameters = ["PDGD", "MDP_positiveDCG", "MDP_negativeDCG", "MDP_pos+neg", "MDP_pos+neg_naive",
                  "MDP_negativeDCG_naive", "MDP_positiveDCG_naive", "MDP_pos+neg_prop1.5"]
    # parameters = ["PDGD", "MDP_negativeDCG", "MDP_pos+neg_naive", "MDP_negativeDCG_naive", ]
    num_interactions = 100000

    plt.figure(1)
    for plot_ind, click_model in enumerate(click_models):
        plot(path1, folds, runs, click_model, num_interactions, 7, plot_ind)
        plot(path2, folds, runs, click_model, num_interactions, 3, plot_ind)
        plot(path3, folds, runs, click_model, num_interactions, 1, plot_ind)
        plot(path4, folds, runs, click_model, num_interactions, 0, plot_ind)
        plot(path5, folds, runs, click_model, num_interactions, 2, plot_ind)
        plot(path6, folds, runs, click_model, num_interactions, 5, plot_ind)
        plot(path7, folds, runs, click_model, num_interactions, 6, plot_ind)
        plot(path8, folds, runs, click_model, num_interactions, 4, plot_ind)
        print()

    plt.legend(parameters, loc='lower right')
    plt.show()
