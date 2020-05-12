import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, t

COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

def plot(path, parameter, folds, runs, click_model, num_interactions, color):
    color_index = 0
    result = np.zeros(num_interactions)
    for f in folds:
        for r in runs:
            with open("{}/fold{}/{}_run{}_cndcg.txt".format(path, f, click_model, r),
                      "rb") as fp:
                data = pickle.load(fp)
                data = np.array(data[:num_interactions])
                result = np.vstack((result, data))
    result = result[1:].T
    result_mean = np.mean(result, axis=1)
    result_std_err = sem(result, axis=1)
    result_h = result_std_err * t.ppf((1 + 0.95) / 2, 25 - 1)
    result_low = np.subtract(result_mean, result_h)
    result_high = np.add(result_mean, result_h)

    plt.plot(range(num_interactions), result_mean, color=COLORS[color], alpha=1)
    # plt.fill_between(range(num_interactions), result_low, result_high, color='black', alpha=0.2)
    color_index += 1
    cndcg = 0
    for i in range(0, len(result_mean) + 1):
        cndcg += 0.9995 ** i * result_mean[i - 1]
    print(parameter, cndcg)

    plt.figure(1)



if __name__ == "__main__":
    path1 = "results/mslr10k/PDGD"
    path2 = "results/mslr10k/MDP_unbiased"
    # path1 = "results/mq2007/PDGD"
    # path2 = "results/mq2007/MDP_unbiased"
    folds = list(range(1, 2))
    runs = list(range(1, 7))
    click_model = 'informational'
    # parameters = [0.03, 0.05, 0.08, 0.1, 0.5, 1.0, 5.0]
    parameters = ["PDGD", "MDP"]
    num_interactions = 10000

    plot(path1, "PDGD", folds, runs, click_model, num_interactions, 1)
    plot(path2, "MDP", folds, runs, click_model, num_interactions, 2)
    plt.ylabel('NDCG')
    plt.xlabel('EPOCH')
    plt.legend(parameters, loc='lower right')
    # plt.show()