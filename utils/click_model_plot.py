import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, t, ttest_ind
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k']


def read_seen_result_file(simulator, click_model, id):
    path = "../click_model_results/{}/seen_set{}_{}_result.txt".format(simulator, id, click_model)
    f = open(path, "r")
    f.readline()
    perplexities = [[], [], [], []]
    MSEs = [[], [], [], []]
    i = 0
    for line in f:
        perplexities[i].append(list(map(float, line.split(" ")[3:])))
        MSEs[i].append(list(map(float, f.readline().split(" ")[3:])))
        i += 1

    return perplexities, MSEs

def read_unseen_result_file(simulator, click_model, id):
    path = "../click_model_results/{}/unseen_set{}_{}_result.txt".format(simulator, id, click_model)
    f = open(path, "r")
    f.readline()
    for line in f:
        perplexity = list(map(float, line.split(" ")[1:]))
        MSE = list(map(float, f.readline().split(" ")[1:]))

    return perplexity, MSE

def plot_perplexity_MSE_for_unseen_queries(simulator, click_models, p1, p2):
    color_index = 0
    print(simulator)
    for click_model in click_models:
        perplexities = []
        MSEs = []
        perplexity, MSE = read_unseen_result_file(simulator, click_model, 1)
        perplexities.append(perplexity)
        MSEs.append(MSE)

        for id in range(2, 9):
            perplexity, MSE = read_unseen_result_file(simulator, click_model, id)

            perplexities.append(perplexity)
            MSEs.append(MSE)

        perplexities = np.array(perplexities)
        # MSEs = np.array(MSEs)
        MSEs = np.sqrt(MSEs)

        mse_mean = np.mean(MSEs.T, axis=1)
        mse_std_err = sem(MSEs.T, axis=1)
        mse_h = mse_std_err * t.ppf((1 + 0.95) / 2, 25 - 1)
        mse_low = np.subtract(mse_mean, mse_h)
        mse_high = np.add(mse_mean, mse_h)

        perp_mean = np.mean(perplexities.T, axis=1)
        perp_std_err = sem(perplexities.T, axis=1)
        perp_h = perp_std_err * t.ppf((1 + 0.95) / 2, 25 - 1)
        perp_low = np.subtract(perp_mean, perp_h)
        perp_high = np.add(perp_mean, perp_h)


        p1.plot(range(1, 11), mse_mean, color=COLORS[color_index], alpha=1)
        p1.fill_between(range(1, 11), mse_low, mse_high, color=COLORS[color_index], alpha=0.2)
        p1.set_ylabel('RMSE')
        p1.set_ylim([0.02, 0.3])
        p1.set_xlim([1, 10])


        p2.plot(range(1, 11), perp_mean, color=COLORS[color_index], alpha=1)
        p2.fill_between(range(1, 11), perp_low, perp_high, color=COLORS[color_index], alpha=0.2)
        p2.set_ylabel('Perplexity')
        p2.set_ylim([1, 2.2])
        p2.set_xlim([1, 10])
        color_index += 1

        print(click_model, np.sum(mse_mean) / 10, np.sum(perp_mean) / 10)

    return perplexities, MSEs




def plot_perplexity_MSE_for_each_rank(simulator, click_model, p1, p2):

    avg_perplexities, avg_MSEs = read_seen_result_file(simulator, click_model, 1)
    mse_runs = []
    perp_runs = []

    for id in range(2, 9):
        perplexities, MSEs = read_seen_result_file(simulator, click_model, id)
        for i in range(4):
            avg_perplexities[i].append(perplexities[i][0])
            avg_MSEs[i].append(MSEs[i][0])

    avg_perplexities = np.array(avg_perplexities)
    # avg_MSEs = np.array(avg_MSEs)
    avg_MSEs = np.sqrt(avg_MSEs)
    for i in range(0, 4):

        print()
        mse_runs.append(np.mean(avg_MSEs[i], axis=1))
        perp_runs.append(np.mean(avg_perplexities[i], axis=1))

        mse_mean = np.mean(avg_MSEs[i].T, axis=1)
        mse_std_err = sem(avg_MSEs[i].T, axis=1)
        mse_h = mse_std_err * t.ppf((1 + 0.95) / 2, 25 - 1)
        mse_low = np.subtract(mse_mean, mse_h)
        mse_high = np.add(mse_mean, mse_h)

        perp_mean = np.mean(avg_perplexities[i].T, axis=1)
        perp_std_err = sem(avg_perplexities[i].T, axis=1)
        perp_h = perp_std_err * t.ppf((1 + 0.95) / 2, 25 - 1)
        perp_low = np.subtract(perp_mean, perp_h)
        perp_high = np.add(perp_mean, perp_h)

        p1.plot(range(1, 11), mse_mean, color=COLORS[i], alpha=1)
        p1.fill_between(range(1, 11), mse_low, mse_high, color=COLORS[i], alpha=0.2)

        if click_model == "FBNCM":
            click_model = "F-NCM"
        if click_model == "NCM":
            click_model = "DR-NCM"

        p1.set_title("click model: {}".format(click_model))
        p1.set_ylabel('RMSE')
        p1.set_ylim([0, 0.3])
        p1.set_xlim([1, 10])

        p2.plot(range(1, 11), perp_mean, color=COLORS[i], alpha=1)
        p2.fill_between(range(1, 11), perp_low, perp_high, color=COLORS[i], alpha=0.2)
        p2.set_title("click model: {}".format(click_model))
        p2.set_ylabel('Perplexity')
        p2.set_ylim([1, 2])
        p1.set_xlim([1, 10])
        print(click_model, np.sum(mse_mean)/10, click_model, np.sum(perp_mean)/10)
    print("MSE t test")
    for x in range(len(mse_runs)):
        for y in range(x, len(mse_runs)):
            print(x, y, ttest_ind(mse_runs[x], mse_runs[y]))

    print()
    print("Perplexity t test")
    for x in range(len(perp_runs)):
        for y in range(x, len(perp_runs)):
            print(x, y, ttest_ind(perp_runs[x], perp_runs[y]))

    return avg_perplexities, avg_MSEs

def plot_for_each_simulator(simulator, click_models, p1, p2):

    mse_runs = []
    perp_runs = []
    color_index = 0
    print()
    print(simulator)
    for click_model in click_models:
        avg_perplexities, avg_MSEs = read_seen_result_file(simulator, click_model, 1)

        for id in range(2, 9):
            perplexities, MSEs = read_seen_result_file(simulator, click_model, id)
            for i in range(4):
                avg_perplexities[i].append(perplexities[i][0])
                avg_MSEs[i].append(MSEs[i][0])


        avg_perplexities = np.array(avg_perplexities)
        avg_MSEs = np.array(avg_MSEs)
        # avg_MSEs = np.sqrt(avg_MSEs)


        num_runs = avg_perplexities.shape[1]
        num_freq = avg_perplexities.shape[0]


        model_perplexity = np.zeros((num_runs, 10))
        model_MSE = np.zeros((num_runs, 10))

        for i in range(0, 4):
            model_perplexity += avg_perplexities[i]
            model_MSE += np.sqrt(avg_MSEs[i])


        model_perplexity = model_perplexity/num_freq
        model_MSE = model_MSE / num_freq
        # model_MSE = np.sqrt(model_MSE)
        mse_runs.append(np.mean(model_MSE, axis=1))
        perp_runs.append(np.mean(model_perplexity, axis=1))

        mse_mean = np.mean(model_MSE.T, axis=1)
        mse_std_err = sem(model_MSE.T, axis=1)
        mse_h = mse_std_err * t.ppf((1 + 0.95) / 2, 25 - 1)
        mse_low = np.subtract(mse_mean, mse_h)
        mse_high = np.add(mse_mean, mse_h)

        perp_mean = np.mean(model_perplexity.T, axis=1)
        perp_std_err = sem(model_perplexity.T, axis=1)
        perp_h = perp_std_err * t.ppf((1 + 0.95) / 2, 25 - 1)
        perp_low = np.subtract(perp_mean, perp_h)
        perp_high = np.add(perp_mean, perp_h)

        print( click_model, np.sum(mse_mean) / 10, np.sum(perp_mean) / 10)
        if simulator == click_model:
            p1.plot(range(1, 11), mse_mean, color=COLORS[color_index], alpha=1, linestyle='dashed', marker='x')
        else:
            p1.plot(range(1, 11), mse_mean, color=COLORS[color_index], alpha=1,)
        p1.fill_between(range(1, 11), mse_low, mse_high, color=COLORS[color_index], alpha=0.2)
        p1.set_ylabel('RMSE')
        p1.set_ylim([0.02, 0.3])
        p1.set_xlim([1, 10])

        if simulator == click_model:
            p2.plot(range(1, 11), perp_mean, color=COLORS[color_index], alpha=1, linestyle='dashed', marker='x')
        else:
            p2.plot(range(1, 11), perp_mean, color=COLORS[color_index], alpha=1,)

        p2.fill_between(range(1, 11), perp_low, perp_high, color=COLORS[color_index], alpha=0.2)
        p2.set_ylabel('Perplexity')
        p2.set_ylim([1, 2.2])
        p2.set_xlim([1, 10])
        color_index += 1


    print("MSE t test")
    for x in range(len(mse_runs)):
        for y in range(x, len(mse_runs)):
            print(click_models[x], click_models[y], ttest_ind(mse_runs[x],mse_runs[y]))

    print()
    print("Perplexity t test")
    for x in range(len(perp_runs)):
        for y in range(x, len(perp_runs)):
            print(click_models[x], click_models[y], ttest_ind(perp_runs[x],perp_runs[y]))



if __name__ == "__main__":

    simulators = [
        "DCTR",
        "SDBN",
        "UBM",
        'SDBN_reverse'
    ]
    click_models = [
        "FBNCM",
        # 'NCM',
        # "SDBN",
        # 'DCTR',
        # 'UBM',
        # 'SDBN_reverse',
        'RCM',
        'RCTR'
    ]

    #
    # for s in simulators:
    #     f = plt.figure(1, figsize=(13, 12))
    #     # f.suptitle("simulator: {}.".format(s))
    #     plot_index = 1
    #     for cm in click_models:
    #         p1 = plt.subplot(len(click_models), 2, plot_index)
    #         p2 = plt.subplot(len(click_models), 2, plot_index + 1)
    #         if plot_index != len(click_models) * 2 - 1:
    #             plt.setp(p1.get_xticklabels(), visible=False)
    #             plt.setp(p2.get_xticklabels(), visible=False)
    #         avg_perplexities, avg_MSEs = plot_perplexity_MSE_for_each_rank(s, cm, p1, p2)
    #         plot_index += 2
    #     # p1.legend(['10', '100', '1000', '10000'], loc='upper right')
    #     p2.legend(['10', '100', '1000', '10000'], loc='upper right')
    #     p1.set_xlabel('rank')
    #     p2.set_xlabel('rank')
    #     plt.savefig('different_freq.png', bbox_inches='tight')
    #     plt.show()


    # f = plt.figure(1, figsize=(13, 8))
    # plot_index = 1
    #
    # for s in simulators:
    #     p1 = plt.subplot(len(simulators), 2, plot_index)
    #     p2 = plt.subplot(len(simulators), 2, plot_index + 1)
    #     if plot_index != len(simulators)*2 - 1:
    #         plt.setp(p1.get_xticklabels(), visible=False)
    #         plt.setp(p2.get_xticklabels(), visible=False)
    #     p1.set_title("generator: " + s)
    #     p2.set_title("generator: " + s)
    #
    #     plot_for_each_simulator(s, click_models, p1, p2)
    #     plot_index += 2
    #
    # labels = []
    # for cm in click_models:
    #     if cm == "FBNCM":
    #         cm = "F-NCM"
    #     if cm == "NCM":
    #         cm = "DR-NCM"
    #     labels.append(cm)
    #
    # custom_lines = []
    # for i in range(len(click_models)):
    #     custom_lines.append(Line2D([0], [0], color=COLORS[i]))
    # # custom_lines = [Line2D([0], [0], color=cmap(0.), lw=4),
    # #                 Line2D([0], [0], color=cmap(.5), lw=4),
    # #                 Line2D([0], [0], color=cmap(1.), lw=4)]
    #
    # p1.legend(custom_lines, labels, loc='upper left', ncol=3, fontsize='small')
    # # p2.legend(click_models, loc='upper right')
    #
    #
    # p1.set_xlabel('rank')
    # p2.set_xlabel('rank')
    # f.subplots_adjust(wspace=0.3, hspace=0.3)
    #
    # plt.savefig('model_generators.png', bbox_inches='tight')
    # plt.show()


    f = plt.figure(1, figsize=(13, 8))
    plot_index = 1
    for s in simulators:
        p1 = plt.subplot(len(simulators), 2, plot_index)
        p2 = plt.subplot(len(simulators), 2, plot_index + 1)
        if plot_index != len(simulators) * 2 - 1:
            plt.setp(p1.get_xticklabels(), visible=False)
            plt.setp(p2.get_xticklabels(), visible=False)
        p1.set_title("generator: " + s)
        p2.set_title("generator: " + s)

        plot_perplexity_MSE_for_unseen_queries(s, click_models, p1, p2)
        plot_index += 2

    labels = []
    for cm in click_models:
        if cm == "FBNCM":
            cm = "F-NCM"
        if cm == "NCM":
            cm = "DR-NCM"
        labels.append(cm)

    custom_lines = []
    for i in range(len(click_models)):
        custom_lines.append(Line2D([0], [0], color=COLORS[i]))
    # custom_lines = [Line2D([0], [0], color=cmap(0.), lw=4),
    #                 Line2D([0], [0], color=cmap(.5), lw=4),
    #                 Line2D([0], [0], color=cmap(1.), lw=4)]


    p1.legend(custom_lines, labels, loc='upper left')
    # p2.legend(click_models, loc='upper right')


    p1.set_xlabel('rank')
    p2.set_xlabel('rank')
    f.subplots_adjust(wspace=0.3, hspace=0.3)

    plt.savefig('unseen_queries.png', bbox_inches='tight')
    plt.show()
