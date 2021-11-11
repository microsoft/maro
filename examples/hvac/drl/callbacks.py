import numpy as np
import os

import matplotlib.pyplot as plt


def visualize_returns(returns, log_dir, title):
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    x = list(range(len(returns)))
    ax.plot(x, returns, color="red")

    ax.set_xlabel('Episode')
    ax.set_xlim([0, x[-1]])
    ax.set_ylabel('Return')
    ax.set_ylim([np.min(returns), np.max(returns)])

    ax.set_title(title)
    plt.savefig(os.path.join(log_dir, f"{title}.png"))
    plt.clf()
