import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, t

COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

def plot(path, parameters, folds, runs, click_model, num_interactions):
    color_index = 0
    for p in parameters:
        result = np.zeros(num_interactions)
        for f in folds:
            for r in runs:
                with open("{}/fold{}/{}_tau{}_run{}_cndcg.txt".format(path, f, click_model, p, r),
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

        plt.plot(range(num_interactions), result_mean, color=COLORS[color_index], alpha=1)
        # plt.fill_between(range(num_interactions), result_low, result_high, color='black', alpha=0.2)
        color_index += 1
        cndcg = 0
        for i in range(0, len(result_mean) + 1):
            cndcg += 0.9995 ** i * data[i - 1]
        print(p, cndcg)

    plt.figure(1)

    plt.ylabel('NDCG')
    plt.xlabel('EPOCH')
    plt.legend(parameters, loc='lower right')
    plt.show()

if __name__ == "__main__":
    path = "../results/exploration/mq2007/PDGD"
    folds = list(range(1, 6))
    runs = list(range(1, 26))
    click_models = ['informational']
    parameters = [0.1, 0.5, 1.0, 5.0, 10.0]
    num_interactions = 10000

    plot(path, parameters, folds, runs, 'informational', num_interactions)