import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, t
from utils.evl_tool import ttest

COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k']


def compute_onlineNDCG(path, folds, runs, click_model, num_interactions, interval):

    start, end = interval
    result = np.zeros(num_interactions*1000)
    for f in folds:
        for r in runs:
            with open("{}/fold{}/{}_run{}_cndcg.txt".format(path, f, click_model, r),
                      "rb") as fp:
                data = pickle.load(fp)
                data = np.array(data[:num_interactions*1000])
                result = np.vstack((result, data))

    result = result[1:].T

    cndcgs = []
    cndcg = 0
    for i in range(start*1000, end*1000):
        cndcg += result[i]
    cndcgs.append(cndcg / (end*1000 - start*1000))
    print(interval, np.mean(cndcgs))
    return cndcg / (end*1000 - start*1000)

def compute_mertic(path, target_path, folds, runs, click_model, num_interactions, num_change, interval, change_point=0):
    result = 0
    drop = 0
    for f in folds:
        for r in runs:
            with open("{}/fold{}/{}_run{}_ndcg.txt".format(path, f, click_model, r),
                      "rb") as fp:
                data1 = pickle.load(fp)
                data1 = np.array(data1[:num_interactions+num_change])

            with open("{}/fold{}/{}_run{}_ndcg.txt".format(target_path, f, click_model, r),
                      "rb") as fp:
                data2 = pickle.load(fp)
                data2 = np.array(data2[:num_interactions])

            difference = data2[interval[0]: interval[1]] - data1[interval[0]+change_point: interval[1]+change_point]
            drop += ((data1[interval[0]+change_point-1] - data1[interval[0]+change_point]).clip(min=0))/data1[interval[0]+change_point-1]


            difference = difference.clip(min=0)/(data2[interval[0]: interval[1]])
            delta = np.sum(difference)/(interval[1] - interval[0])

            result += delta
    print("drop:", drop/len(runs))
    # print("delta:", result/len(runs))


def plot(path, folds, runs, click_model, num_interactions, num_change, color, plot_ind, interval=None):
    plt.subplot(1, 3, plot_ind + 1)

    if interval is None:
        num_interactions += num_change
    result = np.zeros(num_interactions)

    for f in folds:
        for r in runs:
            with open("{}/fold{}/{}_run{}_ndcg.txt".format(path, f, click_model, r),
                      "rb") as fp:
                data = pickle.load(fp)
                data = np.array(data[:num_interactions])
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

        for i in range(num_interactions-num_change):

            if i % ((num_interactions-num_change)/(num_change+1)) == 0 and i > 0:
                xs.append(i * 1000)
                xs.append(i * 1000)
                i += 1
            else:
                xs.append(i * 1000)
        plt.plot(xs, result_mean, color=COLORS[color], alpha=1)
        plt.fill_between(xs, result_low, result_high, color=COLORS[color], alpha=0.2)


def plot_slots(path, fixed_paths, folds, runs, click_model, num_interactions, num_change, color, plot_ind):

    ###### different intent change patterns#####
    # abrupt = "results/SDBN/PDGD/abrupt_group_change_50k/current_intent"
    # smooth = "results/SDBN/PDGD/abrupt_smooth_group_change_50k/current_intent"
    # leaking = "results/SDBN/PDGD/group_leaking_change_50k/current_intent"
    # fixed1 = "results/SDBN/PDGD/group_fixed_200k/group1"
    # fixed2 = "results/SDBN/PDGD/group_fixed_200k/group2"
    # fixed3 = "results/SDBN/PDGD/group_fixed_200k/group3"
    # fixed4 = "results/SDBN/PDGD/group_fixed_200k/group4"
    # print(click_model)
    #
    # print("intent1")
    # compute_mertic(abrupt, fixed1, folds, runs, click_model, num_interactions, num_change, interval=(0,50))
    # compute_mertic(smooth, fixed1, folds, runs, click_model, num_interactions, num_change, interval=(0, 50))
    # compute_mertic(leaking, fixed1, folds, runs, click_model, num_interactions, num_change, interval=(0, 50))
    # print("intent2")
    # compute_mertic(abrupt, fixed2, folds, runs, click_model, num_interactions, num_change, interval=(50, 100))
    # compute_mertic(smooth, fixed2, folds, runs, click_model, num_interactions, num_change, interval=(50, 100))
    # compute_mertic(leaking, fixed2, folds, runs, click_model, num_interactions, num_change, interval=(50, 100))
    #
    # print("intent3")
    # compute_mertic(abrupt, fixed3, folds, runs, click_model, num_interactions, num_change, interval=(100, 150))
    # compute_mertic(smooth, fixed3, folds, runs, click_model, num_interactions, num_change, interval=(100, 150))
    # compute_mertic(leaking, fixed3, folds, runs, click_model, num_interactions, num_change, interval=(100, 150))
    #
    # print("intent4")
    # compute_mertic(abrupt, fixed4, folds, runs, click_model, num_interactions, num_change, interval=(150, 200))
    # compute_mertic(smooth, fixed4, folds, runs, click_model, num_interactions, num_change, interval=(150, 200))
    # compute_mertic(leaking, fixed4, folds, runs, click_model, num_interactions, num_change, interval=(150, 200))
    # print()

    ###### OLTR algorithms experiments#####
    PMGD = "results/SDBN/PMGD/abrupt_group_change_50k/current_intent"
    COLTR = "results/SDBN/COLTR/abrupt_group_change_lrdecay_50k/current_intent"
    PDGD = "results/SDBN/PDGD/abrupt_group_change_50k/current_intent"
    fixed1 = "results/SDBN/PDGD/group_fixed_200k/group1"
    #
    plot(PMGD, folds, runs, click_model, num_interactions, num_change, color[6], plot_ind)
    plot(COLTR, folds, runs, click_model, num_interactions, num_change, color[5], plot_ind)
    plot(PDGD, folds, runs, click_model, num_interactions, num_change, color[0], plot_ind)
    #
    print(click_model)
    print("DBGD")
    compute_mertic(PMGD, fixed1, folds, runs, click_model, num_interactions, num_change, interval=(50, 100), change_point=1)
    compute_mertic(PMGD, fixed1, folds, runs, click_model, num_interactions, num_change, interval=(100, 150), change_point=2)
    compute_mertic(PMGD, fixed1, folds, runs, click_model, num_interactions, num_change, interval=(150, 200),  change_point=3)
    print("COLTR")
    compute_mertic(COLTR, fixed1, folds, runs, click_model, num_interactions, num_change, interval=(50, 100), change_point=1)
    compute_mertic(COLTR, fixed1, folds, runs, click_model, num_interactions, num_change, interval=(100, 150), change_point=2)
    compute_mertic(COLTR, fixed1, folds, runs, click_model, num_interactions, num_change, interval=(150, 200), change_point=3)
    print("PDGD")
    compute_mertic(PDGD, fixed1, folds, runs, click_model, num_interactions, num_change, interval=(50, 100), change_point=1)
    compute_mertic(PDGD, fixed1, folds, runs, click_model, num_interactions, num_change, interval=(100, 150), change_point=2)
    compute_mertic(PDGD, fixed1, folds, runs, click_model, num_interactions, num_change, interval=(150, 200), change_point=3)

    # print(click_model)
    # print("intent1")
    # l2 = compute_onlineNDCG(PMGD, folds, runs, click_model, num_interactions, (0,50))
    # l1 = compute_onlineNDCG(COLTR, folds, runs, click_model, num_interactions, (0,50))
    # l = compute_onlineNDCG(PDGD, folds, runs, click_model, num_interactions, (0, 50))
    #
    # print(ttest(l, l1), ttest(l, l2))
    #
    # print("intent2")
    # l2 = compute_onlineNDCG(PMGD, folds, runs, click_model, num_interactions, (50, 100))
    # l1 = compute_onlineNDCG(COLTR, folds, runs, click_model, num_interactions, (50, 100))
    # l = compute_onlineNDCG(PDGD, folds, runs, click_model, num_interactions, (50, 100))
    #
    # print(ttest(l, l1), ttest(l, l2))
    #
    # print("intent3")
    # l2 = compute_onlineNDCG(PMGD, folds, runs, click_model, num_interactions, (100, 150))
    # l1 = compute_onlineNDCG(COLTR, folds, runs, click_model, num_interactions, (100, 150))
    # l = compute_onlineNDCG(PDGD, folds, runs, click_model, num_interactions, (100, 150))
    # print(ttest(l, l1), ttest(l, l2))
    #
    # print("intent4")
    # l2 = compute_onlineNDCG(PMGD, folds, runs, click_model, num_interactions, (150, 200))
    # l1 = compute_onlineNDCG(COLTR, folds, runs, click_model, num_interactions, (150, 200))
    # l = compute_onlineNDCG(PDGD, folds, runs, click_model, num_interactions, (150, 200))
    # print(ttest(l, l1), ttest(l, l2))


    ###### plot longterm changeback experiments#####
    # path1 = "results/SDBN/PDGD/abrupt_group_changeback_500k/current_intent"
    # path2 = "results/SDBN/PDGD/group_fixed_1500k/group1"
    # path3 = "results/SDBN/PDGD/group_fixed_1500k/group2"
    # plot(path1, folds, runs, click_model, num_interactions, num_change, color[0], plot_ind)
    # plot(path2, folds, runs, click_model, num_interactions, num_change, color[1], plot_ind, interval=(0, 500))
    # plot(path3, folds, runs, click_model, num_interactions, num_change, color[2], plot_ind, interval=(500, 1000))
    # plot(path2, folds, runs, click_model, num_interactions, num_change, color[1], plot_ind, interval=(1000, 1500))
    #
    #
    # compute_mertic(path1, path2, folds, runs, click_model, num_interactions, num_change, interval=(0, 500))
    # compute_mertic(path1, path3, folds, runs, click_model, num_interactions, num_change, interval=(500, 1000))
    # compute_mertic(path1, path2, folds, runs, click_model, num_interactions, num_change, interval=(1000, 1500))
    # print()


    ###### plot mixed intents experiments#####
    # path1 = "results/SDBN/PDGD/group_mixed_2m/mixed_group"
    # path2 = "results/SDBN/PDGD/group_mixed_2m/group_aware"
    # plot(path1, folds, runs, click_model, num_interactions, num_change, color[0], plot_ind)
    # plot(path2, folds, runs, click_model, num_interactions, num_change, color[1], plot_ind)
    #
    # l2 = compute_onlineNDCG(path1, folds, runs, click_model, num_interactions, (0, 2000))
    # l1 = compute_onlineNDCG(path2, folds, runs, click_model, num_interactions, (0, 2000))
    # print(ttest(l1, l2))

    ###### plot nerual experiments#####
    # path1 = "results/SDBN/deepPDGD/abrupt_group_change_50k/current_intent"
    # path2 = "results/SDBN/PDGD/abrupt_group_change_50k/current_intent"
    # fixed1 = "results/SDBN/deepPDGD/group_fixed_200k/group1"
    # fixed2 = "results/SDBN/deepPDGD/group_fixed_200k/group2"
    # fixed3 = "results/SDBN/deepPDGD/group_fixed_200k/group3"
    # fixed4 = "results/SDBN/deepPDGD/group_fixed_200k/group4"
    # plot(path1, folds, runs, click_model, num_interactions, num_change, color[0], plot_ind)
    # # plot(path2, folds, runs, click_model, num_interactions, num_change, color[1], plot_ind)
    # plot(fixed1, folds, runs, click_model, num_interactions, num_change, color[1], plot_ind, interval=(0, 50))
    # plot(fixed2, folds, runs, click_model, num_interactions, num_change, color[2], plot_ind, interval=(50, 100))
    # plot(fixed3, folds, runs, click_model, num_interactions, num_change, color[3], plot_ind, interval=(100, 150))
    # plot(fixed4, folds, runs, click_model, num_interactions, num_change, color[4], plot_ind, interval=(150, 200))


    ###### plot swap changeback experiments#####
    # path1 = "results/SDBN/PDGD/abrupt_group_changeSwap2_50k/current_intent"
    # path2= "results/SDBN/PDGD/group_fixed_300k/group2_reverse"
    # path3 = "results/SDBN/PDGD/group_fixed_300k/group1"
    # plot(path1, folds, runs, click_model, num_interactions, num_change, color[0], plot_ind)
    # plot(path2, folds, runs, click_model, num_interactions, num_change, color[1], plot_ind, interval=(0, 50))
    # plot(path3, folds, runs, click_model, num_interactions, num_change, color[2], plot_ind, interval=(50, 100))
    # plot(path2, folds, runs, click_model, num_interactions, num_change, color[1], plot_ind, interval=(100, 150))
    # plot(path3, folds, runs, click_model, num_interactions, num_change, color[2], plot_ind, interval=(150, 200))
    # plot(path2, folds, runs, click_model, num_interactions, num_change, color[1], plot_ind, interval=(200, 250))
    # plot(path3, folds, runs, click_model, num_interactions, num_change, color[2], plot_ind, interval=(250, 300))
    #
    # print(click_model)
    # compute_mertic(path1, path2, folds, runs, click_model, num_interactions, num_change, interval=(0, 50),
    #                change_point=0)
    # compute_mertic(path1, path3, folds, runs, click_model, num_interactions, num_change, interval=(50, 100),
    #                change_point=1)
    # compute_mertic(path1, path2, folds, runs, click_model, num_interactions, num_change, interval=(100, 150),
    #                change_point=2)
    # compute_mertic(path1, path3, folds, runs, click_model, num_interactions, num_change, interval=(150, 200),
    #                change_point=3)
    # compute_mertic(path1, path2, folds, runs, click_model, num_interactions, num_change, interval=(200, 250),
    #                change_point=4)
    # compute_mertic(path1, path3, folds, runs, click_model, num_interactions, num_change, interval=(200, 250), change_point=4)
    # compute_mertic(path1, path2, folds, runs, click_model, num_interactions, num_change, interval=(250, 300),
    #                change_point=5)
    # print()

    ###### plot noise experiments#####
    # path_0001 = "results/SDBN/PDGD/abrupt_group_changeback_500k/current_intent"
    # path_0408 = "results/SDBN/PDGD/abrupt_group_changeback0408_500k/current_intent"
    # path_0208 = "results/SDBN/PDGD/abrupt_group_changeback0208_500k/current_intent"
    # path_0406 = "results/SDBN/PDGD/abrupt_group_changeback0406_500k/current_intent"
    # plot(path_0001, folds, runs, 'perfect', num_interactions, num_change, color[0], plot_ind)
    # plot(path_0208, folds, runs, click_model, num_interactions, num_change, color[1], plot_ind)
    # plot(path_0408, folds, runs, click_model, num_interactions, num_change, color[2], plot_ind)
    # plot(path_0406, folds, runs, click_model, num_interactions, num_change, color[3], plot_ind)

    # path_fixed_0406_1 = "results/SDBN/PDGD/group_fixed_0406_1500k/group1"
    # path_fixed_0406_2 = "results/SDBN/PDGD/group_fixed_0406_1500k/group2"
    # plot(path_fixed_0406_1, folds, runs, click_model, num_interactions, num_change, color[4], plot_ind, interval=(0, 500))
    # plot(path_fixed_0406_2, folds, runs, click_model, num_interactions, num_change, color[5], plot_ind, interval=(500, 1000))
    # plot(path_fixed_0406_1, folds, runs, click_model, num_interactions, num_change, color[4], plot_ind, interval=(1000, 1500))

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

        plt.legend([
                    # "current_intent",
                    # "fixed_intent1",
                    # "fixed_intent2",
                    # "fixed_intent3",
                    # "fixed_intent4",
                    # "PDGD_Neural",
                    # "PDGD_Linear"
                    "mixed_intents",
                    "intent_specific"
                    # "perfect",
                    # 'small noise',
                    # 'large noise',
                    # 'near random'

                    ], loc='lower right', ncol=3)

    else:
        plt.yticks([])
    # ax = plt.twiny()
    #
    # xticks = []
    # for x in range(num_change+1):
    #     xticks.append((x + 1) * ((num_interactions/(num_change+1))/2) * 1000
    #                   + (((num_interactions/(num_change+1))/2) * 1000 * x))

    # ax.set_xticks(xticks)
    # ax.set_xticklabels(["intent1", "intent2", "intent1", "intent4", "intent1", "intent2"])
    # ax.set_xticklabels(["intent1", "intent2", "intent1", "intent2", "intent1", "intent2"])
    # ax.set_xlim(0, num_interactions*1000)
    plt.tight_layout()


if __name__ == "__main__":

    folds = list(range(1, 2))
    runs = list(range(1, 26))
    # num_interactions = 1500
    num_interactions = 200
    num_change = 3


    click_models = ['perfect', 'navigational', 'informational']
    # click_models = ["noisy"]
    # click_models = ["perfect"]
    plt.figure(1, figsize=(18, 3.5))

    for i in range(len(click_models)):
        plot_slots("path5", "fixed_paths", folds, runs, click_models[i], num_interactions, num_change, [0, 4, 3, 2, 1, 5, 6], i)

    # plt.show()
    # plt.savefig('plots/PDGD_mixed.png', bbox_inches='tight')