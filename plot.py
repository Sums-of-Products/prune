import numpy as np
import matplotlib.pyplot as plt
import os
import re

def read_data(score_name):
    pattern = re.compile(f"pruning-stats-{re.escape(score_name)}-.*")
    data_files = [f for f in os.listdir('data/res') if pattern.match(f)]
    all_data = []

    for file_name in data_files:
        file_path = os.path.join('data/res', file_name)
        data = np.load(file_path)
        specific_part = file_name.replace(f"pruning-stats-{score_name}-", '').replace('.npy', '')
        all_data.append((data, specific_part))

    return all_data

def plot_data(score_name):
    all_data = read_data(score_name)
    if not all_data:
        print("No data files found.")
        return

    n_plots = len(all_data)
    fig, axs = plt.subplots(1, n_plots, figsize=(n_plots * 5, 5))  # Adjust figsize as needed
    
    if n_plots == 1:
        axs = [axs]  # Make it iterable if there's only one subplot

    all_percentages = []

    for i, (data, specific_part) in enumerate(all_data):
        percentages = data[0]
        epss = data[1]
        all_percentages.extend(percentages)

        axs[i].set_xscale('log')
        axs[i].plot(epss, percentages, label=f'{score_name}-{specific_part}')
        axs[i].scatter(epss, percentages)
        axs[i].set_title(f'{score_name}-{specific_part}')
        axs[i].set_xlabel('eps')
        axs[i].set_ylabel('% of scores')

    all_percentages = np.array(all_percentages)
    global_min, global_max = all_percentages.min() * 0.8, all_percentages.max() * 1.2
    for ax in axs:
        ax.set_ylim(global_min, global_max)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    score_name = 'insurance'  # You can change this to use input() or any specific score name
    plot_data(score_name)
