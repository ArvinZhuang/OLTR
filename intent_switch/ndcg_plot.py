import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, t

COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k']


def plot(path, folds, runs, click_model, num_interactions, color, plot_ind, interval=None):
    plt.subplot(1, 3, plot_ind + 1)
    result = np.zeros(num_interactions+3)
    for f in folds:
        for r in runs:
            with open("{}/fold{}/{}_run{}_ndcg.txt".format(path, f, click_model, r),
                      "rb") as fp:
                data = pickle.load(fp)
                data = np.array(data[:num_interactions+3])
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
            if i % 50 == 0 and i > 0:
                xs.append(i * 1000)
                xs.append(i * 1000)
                i += 1
            else:
                xs.append(i * 1000)
        plt.plot(xs, result_mean, color=COLORS[color], alpha=1)
        plt.fill_between(xs, result_low, result_high, color=COLORS[color], alpha=0.2)


def plot_slots(path, fixed_paths, folds, runs, click_model, num_interactions, color, plot_ind):

    plot(path, folds, runs, click_model, num_interactions, color[0], plot_ind)
    # plot("results/SDBN/PDGD/abrupt_change_1234/current_intent", folds, runs, click_model, num_interactions, color[6], plot_ind)
    plot(fixed_paths[0], folds, runs, click_model, num_interactions, color[1], plot_ind, interval=(0, 50))
    plot(fixed_paths[1], folds, runs, click_model, num_interactions, color[2], plot_ind, interval=(50, 100))
    plot(fixed_paths[2], folds, runs, click_model, num_interactions, color[3], plot_ind, interval=(100, 150))
    plot(fixed_paths[3], folds, runs, click_model, num_interactions, color[4], plot_ind, interval=(150, 200))

    plt.xlim(0, 200000)
    plt.ylim(0.1, 0.5)
    xcoords = [50000, 100000, 150000]
    for xc in xcoords:
        plt.axvline(x=xc, color='black', ls='--')

    plt.xlabel('Impressions')
    plt.gca().set_title(click_model)
    if plot_ind == 0:
        plt.ylabel('NDCG')
        plt.legend(['abrupt_change',
                    "intent1_fixed",
                    "intent2_fixed",
                    "intent3_fixed",
                    "intent4_fixed"], loc='lower right', ncol=3)
    else:
        plt.yticks([])
    ax = plt.twiny()

    ax.set_xticks([25000, 75000, 125000, 175000])
    ax.set_xticklabels(["intent1", "intent2", "intent3", "intent4"])
    ax.set_xlim(0, 200000)
    plt.tight_layout()
# def make_plots(path, fixed_paths, folds, runs, click_model, num_interactions, color):


if __name__ == "__main__":
    fixed_path1 = "results/SDBN/PDGD/group_fixed_200k/group1"
    fixed_path2 = "results/SDBN/PDGD/group_fixed_200k/group2"
    fixed_path3 = "results/SDBN/PDGD/group_fixed_200k/group3"
    fixed_path4 = "results/SDBN/PDGD/group_fixed_200k/group4"
    fixed_paths = [fixed_path1, fixed_path2, fixed_path3, fixed_path4]
    path1 = "results/SDBN/COLTR/intent_leaking_1234/current_intent"
    path2 = "results/SDBN/PDGD/abrupt_change_1234/current_intent"
    path3 = "results/SDBN/PDGD/abrupt_group_change_50k/current_intent"

    folds = list(range(1, 2))
    runs = list(range(1, 2))
    # click_models = ['navigational']
    # parameters = [0.03, 0.05, 0.08, 0.1, 0.5, 1.0, 5.0]
    # num_interactions = 400
    num_interactions = 200

    click_models = ['perfect', 'navigational', 'informational']
    plt.figure(1, figsize=(18, 3.5))

    for i in range(len(click_models)):
        plot_slots(fixed_path2, fixed_paths, folds, runs, click_models[i], num_interactions, [0,4,3,2,1,5,6], i)

    # plt.show()
    # plt.savefig('COLTR_abrupt_smooth_1234.png', bbox_inches='tight')