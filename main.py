from src.pruning import ScorePruner
import matplotlib.pyplot as plt
import numpy as np

from src.utils import read_scores_from_file


score_names = ['insurance-400', 'insurance-600']
# score_names = ['hailfinder-100', 'hailfinder-200', 'hailfinder-500', 'hailfinder-1000']
# score_names = ['hailfinder-2000-p=4', 'hailfinder-6000-p=4']
import numpy as np
import matplotlib.pyplot as plt

# score_names = ['mushrooms-100', 'mushrooms-200', 'mushrooms-500', 'mushrooms-1000', 'mushrooms-2000', 'mushrooms-4000']
epss = np.logspace(-9, -1, 9)
n = len(score_names)

fig, axs = plt.subplots(1, n)

all_percentages = []
# Assuming read_scores_from_file and ScorePruner are defined elsewhere
for i in range(0, n):
    score_name = score_names[i]
    orig_scores = read_scores_from_file(f'data/scores/{score_name}.jkl')
    

    percentages = []
    for eps in epss:
        scores = orig_scores.copy()
        pruner = ScorePruner(scores, 4, eps, 1)  # b is set to 1
        original_count = pruner.count_scores()
        pruner.prune_scores()
        pruned_scores_count = pruner.count_scores()

        # Calculate the percentage of scores left
        percentage_left = round((pruned_scores_count / original_count) * 100, 6)
        percentages.append(percentage_left)
        all_percentages.append(percentage_left)

    axs[i].set_xscale('log')
    axs[i].plot(epss, percentages)
    axs[i].scatter(epss, percentages)
    axs[i].set_title(f'{score_name}')
    axs[i].set_xlabel('Epsilon')
    axs[i].legend()
    axs[i].set_ylabel('Percentage of Scores Left')

all_percentages = np.array(all_percentages)
for ax in axs:
    ax.set_ylim(all_percentages.min() * 0.99, all_percentages.max()*1.01)
plt.legend()
plt.show()

