import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, t
import matplotlib.gridspec as gridspec

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

    for id in range(1,2):
        perplexities, MSEs = read_set_result_file(simulator, click_model, id)
        for i in range(5):
            avg_perplexities[i].append(perplexities[i][0])
            avg_MSEs[i].append(MSEs[i][0])

    avg_perplexities = np.array(avg_perplexities)
    avg_MSEs = np.array(avg_MSEs)
    print("test")
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
        p1.set_title(click_model)
        p1.set_ylabel('MSE')
        p1.set_ylim([0, 0.05])

        p2.plot(range(1, 11), perp_mean, color=COLORS[i], alpha=1)
        p2.fill_between(range(1, 11), perp_low, perp_high, color=COLORS[i], alpha=0.2)
        p2.set_title(click_model)
        p2.set_ylabel('Perplexity')
        p2.set_ylim([1, 2])

    return avg_perplexities, avg_MSEs

def plot_for_each_simulator(simulator, click_models, p1, p2):

    color_index = 0
    for click_model in click_models:
        avg_perplexities, avg_MSEs = read_set_result_file(simulator, click_model, 1)

        for id in range(1, 2):
            perplexities, MSEs = read_set_result_file(simulator, click_model, id)
            for i in range(5):
                avg_perplexities[i].append(perplexities[i][0])
                avg_MSEs[i].append(MSEs[i][0])

        avg_perplexities = np.array(avg_perplexities)
        avg_MSEs = np.array(avg_MSEs)

        num_runs = avg_perplexities.shape[1]
        num_freq = avg_perplexities.shape[0]


        model_perplexity = np.zeros((num_runs, 10))
        model_MSE = np.zeros((num_runs, 10))

        for i in range(num_freq):
            model_perplexity += avg_perplexities[i]
            model_MSE += avg_MSEs[i]
        model_perplexity = model_perplexity/num_freq
        model_MSE = model_MSE / num_freq

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

        p1.plot(range(1, 11), mse_mean, color=COLORS[color_index], alpha=1)
        p1.fill_between(range(1, 11), mse_low, mse_high, color=COLORS[color_index], alpha=0.2)
        p1.set_ylabel('MSE')
        p1.set_ylim([0, 0.04])

        p2.plot(range(1, 11), perp_mean, color=COLORS[color_index], alpha=1)
        p2.fill_between(range(1, 11), perp_low, perp_high, color=COLORS[color_index], alpha=0.2)
        p2.set_ylabel('Perplexity')
        p2.set_ylim([1, 2])
        color_index += 1




if __name__ == "__main__":
    # simulators = ["Mixed"]
    simulators = ["SDBN", 'DCTR', 'UBM', 'SDCM', "Mixed"]
    # click_models = ["SDBN"]
    click_models = ["SDBN", 'DCTR', 'SDCM']

    # for s in simulators:
    #     f = plt.figure(1)
    #     f.suptitle("simulator: {}.".format(s))
    #     plot_index = 1
    #     for cm in click_models:
    #         p1 = plt.subplot(len(click_models), 2, plot_index)
    #         p2 = plt.subplot(len(click_models), 2, plot_index + 1)
    #         avg_perplexities, avg_MSEs = plot_perplexity_MSE_for_each_rank(s, cm, p1, p2)
    #         plot_index += 2
    #     p1.legend(['10', '100', '1000', '10000', '100000'], loc='upper right')
    #     p2.legend(['10', '100', '1000', '10000', '100000'], loc='upper right')
    #     p1.set_xlabel('rank')
    #     p2.set_xlabel('rank')
    #     plt.show()

    f = plt.figure(1)
    plot_index = 1
    for s in simulators:
        p1 = plt.subplot(len(simulators), 2, plot_index)
        p2 = plt.subplot(len(simulators), 2, plot_index + 1)
        p1.set_title("simulator: " + s)
        p2.set_title("simulator: " + s)

        plot_for_each_simulator(s, click_models, p1, p2)
        plot_index += 2

        p1.legend(click_models, loc='upper right')
        p2.legend(click_models, loc='upper right')


    p1.set_xlabel('rank')
    p2.set_xlabel('rank')
    f.subplots_adjust(wspace=0.3, hspace=0.3)

    plt.show()
