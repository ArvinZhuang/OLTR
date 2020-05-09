import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, t

COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

def plot(path, parameters, folds, runs, click_model, num_interactions, color):
    print("result path:", path)
    color_index = 0
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
    result_h = result_std_err * t.ppf((1 + 0.95) / 2, 25 - 1)
    result_low = np.subtract(result_mean, result_h)
    result_high = np.add(result_mean, result_h)

    plt.plot(range(num_interactions), result_mean, color=COLORS[color], alpha=1)

    plt.fill_between(range(num_interactions), result_low, result_high, color=COLORS[color], alpha=0.2)
    color_index += 1
    plt.figure(1)
    plt.legend(parameters, loc='lower right')



if __name__ == "__main__":
    path1 = "results/mq2007/PDGD"
    path2 = "results/mq2007/MDP_unbiased"
    folds = list(range(1, 4))
    runs = list(range(1, 21))
    click_model = 'navigational'
    # parameters = [0.03, 0.05, 0.08, 0.1, 0.5, 1.0, 5.0]
    parameters = ["PDGD", "MDP"]
    num_interactions = 10000

    plot(path1, "PDGD", folds, runs, click_model, num_interactions, 1)
    plot(path2, "MDP", folds, runs, click_model, num_interactions, 2)
    plt.ylabel('NDCG')
    plt.xlabel('EPOCH')
    plt.legend(parameters, loc='lower right')
    plt.show()
