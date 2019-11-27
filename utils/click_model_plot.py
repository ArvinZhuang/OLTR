import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, t

COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k']


def read_set_result_file(simulator, click_model, id):
    path = "../click_model_results/{}/seen_set{}_{}_result.txt".format(simulator, id, click_model)
    f = open(path, "r")
    f.readline()
    perplexities = [[], [], [], [], []]
    MSEs = [[], [], [], [], []]
    i = 0
    for line in f:
        perplexities[i].append(list(map(float, line.split(" ")[3:])))
        MSEs[i].append(list(map(float, f.readline().split(" ")[3:])))
        i += 1

    return perplexities, MSEs


def plot_perplexity_MSE_for_each_rank(simulator, click_model, p1, p2):
    avg_perplexities, avg_MSEs = read_set_result_file(simulator, click_model, 1)

    for id in range(2, 16):
        perplexities, MSEs = read_set_result_file(simulator, click_model, id)
        for i in range(5):
            avg_perplexities[i].append(perplexities[i][0])
            avg_MSEs[i].append(MSEs[i][0])

    avg_perplexities = np.array(avg_perplexities)
    avg_MSEs = np.array(avg_MSEs)

    for i in range(5):
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
        p1.set_title(cm)
        p1.set_ylabel('MSE')
        p1.set_ylim([0, 0.05])

        p2.plot(range(1, 11), perp_mean, color=COLORS[i], alpha=1)
        p2.fill_between(range(1, 11), perp_low, perp_high, color=COLORS[i], alpha=0.2)
        p2.set_title(cm)
        p2.set_ylabel('Perplexity')
        p2.set_ylim([1, 2])

    return avg_perplexities, avg_MSEs


if __name__ == "__main__":
    simulators = ["SDBN", 'SDCM', 'DCTR', 'CM']
    click_models = ["SDBN", 'SDCM', 'DCTR', 'CM']
    for s in simulators:
        f = plt.figure(1)
        f.suptitle("simulator: {}.".format(s))
        plot_index = 1
        for cm in click_models:
            p1 = plt.subplot(len(click_models), 2, plot_index)
            p2 = plt.subplot(len(click_models), 2, plot_index + 1)
            avg_perplexities, avg_MSEs = plot_perplexity_MSE_for_each_rank(s, cm, p1, p2)
            plot_index += 2
        p1.legend(['10', '100', '1000', '10000', '100000'], loc='upper right')
        p2.legend(['10', '100', '1000', '10000', '100000'], loc='upper right')
        p1.set_xlabel('rank')
        p2.set_xlabel('rank')
        plt.show()
