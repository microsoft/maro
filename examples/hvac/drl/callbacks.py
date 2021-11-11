import numpy as np
import os

import matplotlib.pyplot as plt


def visualize_rolling_scores(result, log_dir, title):
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    x = list(range(len(result)))
    ax.plot(x, result, color="red")

    ax.set_xlabel('Episode Number')
    ax.set_xlim([0, x[-1]])
    ax.set_ylabel('Rolling Episode Scores')
    ax.set_ylim([np.min(result), np.max(result)])

    ax.set_title(title)
    plt.savefig(os.path.join(log_dir, f"Rolling score - {title}.png"))
