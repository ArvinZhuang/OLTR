import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, t

COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k']


def plot(path, folds, runs, click_model, num_interactions, num_change, color, plot_ind, interval=None):
    plt.subplot(1, 3, plot_ind + 1)
    if interval is not None:
        result = np.zeros(num_interactions)
    else:
        result = np.zeros(num_interactions+num_change)
    for f in folds:
        for r in runs:
            with open("{}/fold{}/{}_run{}_ndcg.txt".format(path, f, click_model, r),
                      "rb") as fp:
                data = pickle.load(fp)
                data = np.array(data[:num_interactions+num_change])
                result = np.vstack((result, data))
    result = result[1:].T
    result_mean = np.mean(result, axis=1)
    result_std_err = sem(result, axis=1)
    result_h = result_std_err * t.ppf((1 + 0.95) / 2, 16 - 1)
    result_low = np.subtract(result_mean, result_h)
    result_high = np.add(result_mean, result_h)


    if interval is not None:
        xs = []
        for i in range(interval[0], interval[1]):
            xs.append(i * 1000)

        plt.plot(xs, result_mean[interval[0]: interval[1]], color=COLORS[color], alpha=1)
        plt.fill_between(xs, result_low[interval[0]: interval[1]], result_high[interval[0]: interval[1]], color=COLORS[color], alpha=0.2)
    else:
        xs = []
        for i in range(num_interactions):
            if i % (num_interactions/(num_change+1)) == 0 and i > 0:
                xs.append(i * 1000)
                xs.append(i * 1000)
                i += 1
            else:
                xs.append(i * 1000)
        plt.plot(xs, result_mean, color=COLORS[color], alpha=1)
        plt.fill_between(xs, result_low, result_high, color=COLORS[color], alpha=0.2)


def plot_slots(path, fixed_paths, folds, runs, click_model, num_interactions, num_change, color, plot_ind):

    plot(path, folds, runs, click_model, num_interactions, num_change, color[0], plot_ind)

    plot(fixed_paths[0], folds, runs, click_model, num_interactions, num_change, color[1], plot_ind, interval=(0, 50))
    plot(fixed_paths[1], folds, runs, click_model, num_interactions, num_change, color[2], plot_ind, interval=(50, 100))
    # plot(fixed_paths[0], folds, runs, click_model, num_interactions, num_change, color[1], plot_ind, interval=(1000, 1500))
    plot(fixed_paths[2], folds, runs, click_model, num_interactions, num_change, color[3], plot_ind, interval=(100, 150))
    plot(fixed_paths[3], folds, runs, click_model, num_interactions, num_change, color[4], plot_ind, interval=(150, 200))


    # PMGD = "results/SDBN/PMGD/abrupt_group_change_50k/current_intent"
    # COLTR = "results/SDBN/COLTR/abrupt_group_change_50k/current_intent"
    # PDGD = "results/SDBN/PDGD/abrupt_group_change_50k/current_intent"
    # plot(PMGD, folds, runs, click_model, num_interactions, num_change, color[6], plot_ind)
    # plot(COLTR, folds, runs, click_model, num_interactions, num_change, color[5], plot_ind)
    # plot(PDGD, folds, runs, click_model, num_interactions, num_change, color[0], plot_ind)

    # current1 = "results/SDBN/PDGD/abrupt_group_change_50k/intent1"
    # current2 = "results/SDBN/PDGD/abrupt_group_change_50k/intent2"
    # current3 = "results/SDBN/PDGD/abrupt_group_change_50k/intent3"
    # current4 = "results/SDBN/PDGD/abrupt_group_change_50k/intent4"
    # plot(current1, folds, runs, click_model, num_interactions, num_change, color[1], plot_ind, interval=(0, 200))
    # plot(current2, folds, runs, click_model, num_interactions, num_change, color[2], plot_ind, interval=(0, 200))
    # plot(current3, folds, runs, click_model, num_interactions, num_change, color[3], plot_ind, interval=(0, 200))
    # plot(current4, folds, runs, click_model, num_interactions, num_change, color[4], plot_ind, interval=(0, 200))

    plt.xlim(0, num_interactions*1000)
    plt.ylim(0.25, 0.5)
    xcoords = []
    for x in range(num_change):
        xcoords.append((x + 1) * (num_interactions/(num_change+1)) * 1000)
    for xc in xcoords:
        plt.axvline(x=xc, color='black', ls='--')

    plt.xlabel('Impressions')
    plt.gca().set_title(click_model)
    if plot_ind == 0:
        plt.ylabel('NDCG')

        # plt.legend(['DBGD',
        #             "COLTR",
        #             "PDGD"], loc='lower right', ncol=3)

        plt.legend(["current_intent",
                    "intent1_fixed",
                    "intent2_fixed",
                    "intent3_fixed",
                    "intent4_fixed",], loc='lower right', ncol=3)

        # plt.legend(["current_intent1",
        #             "current_intent2",
        #             "current_intent3",
        #             "current_intent4", ], loc='lower right', ncol=3)
    else:
        plt.yticks([])
    ax = plt.twiny()

    xticks = []
    for x in range(num_change+1):
        xticks.append((x + 1) * ((num_interactions/(num_change+1))/2) * 1000
                      + (((num_interactions/(num_change+1))/2) * 1000 * x))

    ax.set_xticks(xticks)
    ax.set_xticklabels(["intent1", "intent2", "intent3", "intent4"])
    # ax.set_xticklabels(["intent1", "intent2", "intent1"])
    ax.set_xlim(0, num_interactions*1000)
    plt.tight_layout()


if __name__ == "__main__":
    fixed_path1 = "results/SDBN/PDGD/group_fixed_1500k/group1"
    fixed_path2 = "results/SDBN/PDGD/group_fixed_1500k/group2"
    fixed_path3 = "results/SDBN/PDGD/group_fixed_200k/group1"
    fixed_path4 = "results/SDBN/PDGD/group_fixed_200k/group2"
    fixed_path5 = "results/SDBN/PDGD/group_fixed_200k/group3"
    fixed_path6 = "results/SDBN/PDGD/group_fixed_200k/group4"
    fixed_paths = [fixed_path3, fixed_path4, fixed_path5, fixed_path6]
    path1 = "results/SDBN/PDGD/abrupt_smooth_group_change_50k/current_intent"
    path2 = "results/SDBN/PDGD/abrupt_group_changeback_500k/current_intent"
    path3 = "results/SDBN/PDGD/group_leaking_change_50k/current_intent"



    folds = list(range(1, 2))
    runs = list(range(1, 2))
    # click_models = ['navigational']
    # parameters = [0.03, 0.05, 0.08, 0.1, 0.5, 1.0, 5.0]
    num_interactions = 200
    # num_interactions = 1500
    num_change = 3


    click_models = ['perfect', 'navigational', 'informational']
    # click_models = ["perfect"]
    plt.figure(1, figsize=(18, 3.5))

    for i in range(len(click_models)):
        plot_slots(path3, fixed_paths, folds, runs, click_models[i], num_interactions, num_change, [0,4,3,2,1,5,6], i)

    # plt.show()
    # plt.savefig('plots/PDGD_each_current_intent_shortterm.png', bbox_inches='tight')