from pruning import count_scores, init_pruning, prune_scores
import matplotlib.pyplot as plt
import copy
import numpy as np

from utils import read_scores_from_file


score_names = ['insurance-400', 'insurance-600', 'insurance-1000']


epss = [0.001, 0.01, 0.1, 0.4, np.e]
n = len(score_names)

all_values = []
fig, axs = plt.subplots(1, n)

for i in range(0, n):
    score_name = score_names[i]
    plt.title(score_name)
    orig_scores = read_scores_from_file(f'data/scores/{score_name}.jkl')

    for j in range(len(epss)):
        eps = epss[j]

        scores = orig_scores.copy()

        pruning_results = []
        bs = np.linspace(0, 1, 7)
        for b in bs:
            pruned_scores = copy.deepcopy(scores)
            init_pruning(pruned_scores, 4, eps, b)

            prune_scores()
            pruned_scores_count = count_scores(pruned_scores)
            pruning_results.append(pruned_scores_count)
            all_values.append(pruned_scores_count)

        axs[i].plot(bs, pruning_results, label=f'eps={eps}')
        axs[i].set_title(f'{score_name}')
        plt.xlabel('b')

all_values = np.array(all_values)
for ax in axs:
    ax.set_ylim(all_values.min() * 0.98, all_values.max()*1.02)

plt.legend()
plt.show()
