import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from utils import (NIDO_RdRp_domain, easy_grouping, )


def postprocessing(results: dict, hmm_data: pd.DataFrame, params: dict):

    # extract relevant parameters
    file_name = params["file"]
    T = params["t"]

    # extract data (y=distances/similarity, x=samples/header)
    y, x = results["y"], results["samples"]

    # prepare figure for plotting
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.grid(axis="y")
    ax2.grid(axis="y")

    # plotting
    ax2.scatter(x, y, color="#832621", s=0.6, zorder=1, label=f"cosine")
    ax1.set_ylim(-1.05, 1.05)
    ax2.set_ylim(-1.05, 1.05)
    ax2.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax1.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax1.set_xticks([])

    # axes titles
    ax2.set_xlabel("Snippets")
    ax2.set_ylabel(f"Similarity")
    ax1.set_ylabel("Groups")

    # get group information
    group_data = easy_grouping(data=y, t=T)

    # unpacking x positions
    x_pos = np.asarray([np.mean(sample) * 1.005 for sample in group_data['s']])

    # find the group with the max number of members and a corresponding high mean similarity
    ph_filter = group_data["ph"] == 1
    phd = group_data["md"][ph_filter]

    if len(phd) > 0:
        max_id = np.argmax(phd)
        max_pos, max_samples = x_pos[ph_filter][max_id], group_data["s"][ph_filter][max_id]
        max_d = group_data["md"][ph_filter][max_id]

    else:
        max_id, max_pos, max_samples, max_d = -1, -1, [], -1

    # plot
    for i, (xp, height, width) in enumerate(zip(x_pos, group_data['md'], group_data['gl'])):
        if i == max_id:
            ax1.bar(xp, height, width=width, align="center", edgecolor=None, color="#d27011", alpha=0.8)

        else:
            ax1.bar(xp, height, width=width, align="center", edgecolor=None, color="#ffe5c5", alpha=0.8)

        if width == 1 and height > 0.0:
            ax1.scatter(xp, height, marker="*", s=0.5, color="#e4645b", zorder=4)

    # add HMM overlay
    hmm_hit_range = NIDO_RdRp_domain(target_header=file_name, hmm_data=hmm_data, metacontigs=params["metacontigs"])
    hmm_left, hmm_right = hmm_hit_range[0], hmm_hit_range[1]

    # plot threshold @ 0.0
    _, x_max, y_min, y_max = ax1.axis()
    ax1.hlines(y=T, xmin=0.0, xmax=x_max, linewidth=0.25, color="#300b0a")

    #
    if hmm_left != hmm_right:

        # add overlay to local plot
        hmm_fig = mpatches.Rectangle((hmm_left, -1), min(hmm_right - 10, x_max - 10), 2 * max(y_max, 1.0), fill=True, zorder=4,
                                     # color="#009999", linewidth=1, alpha=0.25,
                                     color="#666600", linewidth=1, alpha=0.15,
                                     label="Hidden Markov Model (HMM))")
        ax1.add_patch(hmm_fig)

    default_return = [max_pos, max_d, (max_samples), hmm_left, hmm_right]

    # 3 - Save figure
    # ----------------------------------------------------------------------------------------------------------------
    # ax1.legend(loc="upper right", fontsize=7)
    # ax2.legend(loc="upper right", fontsize=7)
    fig.savefig(params["save_to"] + f"/{file_name}_results.svg")
    plt.close()

    return default_return
