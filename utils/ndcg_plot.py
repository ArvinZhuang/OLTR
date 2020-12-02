import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, t

COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

def plot(path, parameters, folds, runs, click_model, num_interactions, color):
    print(path)
    color_index = 0
    for p in parameters:
        result = np.zeros(num_interactions)
        for f in folds:
            for r in runs:
                with open("{}/fold{}/{}_run{}_ndcg.txt".format(path, f, click_model, r),
                          "rb") as fp:
                    data = pickle.load(fp)
                    data = np.array(data[:num_interactions])
                    result = np.vstack((result, data))
        result = result[1:].T
        print(result.shape)
        result_mean = np.mean(result, axis=1)
        result_std_err = sem(result, axis=1)
        result_h = result_std_err * t.ppf((1 + 0.95) / 2, 25 - 1)
        result_low = np.subtract(result_mean, result_h)
        result_high = np.add(result_mean, result_h)

        plt.plot(range(num_interactions), result_mean, color=COLORS[color], alpha=1)

        plt.fill_between(range(num_interactions), result_low, result_high, color=COLORS[color], alpha=0.2)
        color_index += 1


def plot_mrr(path, folds, runs, click_models, num_interactions, color):
    fig, axs = plt.subplots(3, 5)
    row = 0
    column = 0
    for click_model in click_models:
        axs[row, column].set_title(click_model)

        for f in folds:

            for r in runs:
                with open("{}/fold{}/{}_run{}_cndcg.txt".format(path, f, click_model, r),
                          "rb") as fp:
                    data = pickle.load(fp)
                    data = np.array(data)

                axs[row, column].plot(range(num_interactions), [sum(group) / 8000 for group in zip(*[iter(data)]*8000)], color=COLORS[color])
                # if click_model == 'informational':
                #     axs[row, column].set_ylim([0.70, 0.76])
                # elif click_model == 'navigational':
                #     axs[row, column].set_ylim([0.43, 0.52])
                # else:
                #     axs[row, column].set_ylim([0.4, 0.47])
            column += 1
        row += 1
        column = 0


if __name__ == "__main__":
    # path1 = "../results/PDGD/mq2007"
    path1 = "../results/exploration/PDGD/istella/random"
    path2 = "../results/exploration/PDGD/istella/original"
    # path2 = "../results/reduction/mq2007/PDGD"
    folds = list(range(1, 2))
    runs = list(range(1, 2))
    click_models = ["navigational", 'informational', "perfect"]
    # parameters = [0.03, 0.05, 0.08, 0.1, 0.5, 1.0, 5.0]
    parameters = [0.1]
    num_interactions = 1000

    # plot(path1, parameters, folds, runs, 'informational', num_interactions, 1)
    plot(path1, parameters, folds, runs, 'perfect', num_interactions, 2)
    plot(path2, parameters, folds, runs, 'perfect', num_interactions, 1)
    # plot_mrr(path1, folds, runs, click_models, num_interactions, 2)
    plt.ylabel('NDCG')
    plt.xlabel('EPOCH')

    # plt.legend(click_models, loc='lower right')
    plt.show()