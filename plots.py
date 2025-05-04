import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import umap


# function to plot local Hamming score for a given data set
def local_hs_plot(no_samples: list, local_hs: list, set_type: str, save_to: str):

    plt.plot(no_samples, local_hs, label=f"{set_type}", c="peru")
    plt.xlabel("Number Samples")
    plt.xlabel("Hamming Score")

    plt.legend()
    plt.savefig(f"{save_to}_localHS_{set_type}_plot.svg", format="svg")

    print(f"saved local Hamming score plot.")
    plt.close()


# function to plot training results
def train_plot(epochs: list, train_loss: list, save_as: str):

    plt.plot(epochs, train_loss, label="Train", c='khaki')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.legend()
    plt.savefig(save_as, format="svg")
    print(f"saved training plot.")
    plt.close()


def test_plot(results: pd.DataFrame, save_as: str, y_str: str = "Similarity"):

    # unpack
    sim, labels, samples = results["y"].tolist(), results["labels"].tolist(), results["samples"].tolist()
    colors = ["#CC0066", "#0066CC"]

    # init color coding
    color_map = {l: colors[i] for i, l in enumerate(np.unique(labels))}

    # plotting
    for i, label in enumerate(labels):
        plt.scatter(samples[i], sim[i], color=color_map[label], s=1)

    # handles
    handles = [plt.scatter([], [], color=c, label=l) for l, c in color_map.items()]

    plt.axhline(y=0.0, color="#606060", linestyle="--", alpha=0.85)
    plt.ylim(-1.005, 1.005)
    plt.yticks([-1, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
    plt.xlabel("Test Snippets")
    plt.ylabel(y_str)
    plt.grid(True)
    plt.legend(handles, color_map.keys())
    plt.savefig(save_as, format="svg")
    plt.close()


def loss_plotting(save_to: str, losses: list, set_type: str, k: str):

    def __melt_dataframe(df):

        return pd.DataFrame.from_dict(df).reset_index().melt(id_vars=["index"]).rename(columns={"index":"epochs"})

    # plotting loss and accuracy in one figure
    if isinstance(losses, dict):

        # convert to data frames
        loss_df = __melt_dataframe(losses)

        # plotting
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(25, 10))

        sns.lineplot(data=loss_df, x="epochs", y="value", hue="variable", ax=axes).set_title("Train Losses")

        plt.legend([], [], frameon=False)
        plt.tight_layout()
        plt.savefig(save_to + f"losses_plot_0{k}.svg", format="svg")

    print(f"saved loss plots.")


def global_results_boxplot(data: pd.DataFrame, file_name: str, save_to: str):

    fig, ax = plt.subplots()

    labels, scores = data["origin_id"].tolist(), np.asarray(data["score"].tolist())
    label_ids = sorted(list(set(labels)))

    for l_id in label_ids:
        ax.boxplot(scores[np.where(np.asarray(labels) == l_id)[0]], positions=[l_id], widths=0.5, showfliers=False)

    # Customize plot
    ax.set_xticks(label_ids)
    ax.set_xticklabels(label_ids)
    ax.set_xlabel('Label')
    ax.set_ylabel('Score')
    plt.grid(True)

    # saving
    fig.tight_layout()
    fig.savefig(save_to + f"/{file_name}_scores.svg", format="svg")
    plt.close()


def roc_plot(fpr, tpr, roc,save_as: str):

    plt.figure(figsize=(8, 6))
    plt.plot([0, fpr, 1], [0, tpr, 1], color="#009999", lw=2, label="ROC curve (area=%0.2f)" % roc)
    plt.plot([0, 1], [0, 1], color="#000099", lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(save_as, format="svg")
    plt.close()


def global_results_plot(data: pd.DataFrame, file_name: str, save_to: str):

    # set up figure
    fig, ax = plt.subplots()
    ax.scatter(data["header"], data["score"], c=data["origin_id"], marker="o", s=10.0, cmap="Paired")

    # grid
    ax.grid(True)
    ax.grid(color="#404040", linestyle='--', linewidth=0.25)

    # labels
    ax.set_xlabel('')
    ax.set_ylabel('Scores')

    # ticks
    ax.tick_params(axis="x", labelsize=2.5)
    ax.set_xticklabels(data["header"], rotation=45, ha="right")
    ticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ax.set_yticks(ticks)

    # saving
    fig.tight_layout()
    fig.savefig(save_to + f"/{file_name}_scores.svg")
    plt.close()


def umap_plot(embeddings, labels, save_as: str, metric: str = "cosine",
              nn: int = 15, min_dist: float = 0.1, nc: int = 2):

    # init UMAP with desired parameters
    reducer = umap.UMAP(
        n_neighbors=nn,
        min_dist=min_dist,
        n_components=nc,
        metric=metric,
        spread=1.0,
        set_op_mix_ratio=1.0,
        local_connectivity=1
    )

    # fit and transform embeddings
    reduced_embeddings = reducer.fit_transform(embeddings)

    # plot
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, s=2, cmap="Spectral")
    plt.colorbar(scatter, ticks=range(2))
    # plt.title(f"UMAP projection of test embeddings for k={k}")

    plt.savefig(save_as, format="svg")
    plt.close()

