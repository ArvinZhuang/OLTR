import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, t

COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k']


def plot(path, folds, runs, click_model, num_interactions, color):

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

    plt.plot(range(num_interactions), result_mean, color=COLORS[color], alpha=1)

    plt.fill_between(range(num_interactions), result_low, result_high, color=COLORS[color], alpha=0.2)
    plt.figure(1)
    plt.legend("PDGD", loc='lower right')


if __name__ == "__main__":
    # path = "results/SDBN/PDGD/intent_change_exclusive/current_intent"
    path = "results/SDBN/PDGD/intent_change_exclusive/intent4"


    path1 = "results/SDBN/PDGD/intent_change_samll/intent1"
    path2 = "results/SDBN/PDGD/intent_change_samll/intent2"
    path3 = "results/SDBN/PDGD/intent_fixed/intent3"
    path4 = "results/SDBN/PDGD/intent_fixed/intent4"

    folds = list(range(1, 6))
    runs = list(range(1, 16))
    # click_models = ['navigational']
    # parameters = [0.03, 0.05, 0.08, 0.1, 0.5, 1.0, 5.0]
    num_interactions = 500

    plot(path, folds, runs, 'informational', num_interactions, 1)
    plot(path, folds, runs, 'perfect', num_interactions, 2)


    plt.ylabel('NDCG')
    plt.xlabel('EPOCH')
    plt.legend(['informational', 'perfect'], loc='lower right')
    plt.show()