import numpy as np

import networkx as nx
import matplotlib.pyplot as plt

from matplotlib.colors import PowerNorm
from matplotlib.colors import LogNorm
import matplotlib.ticker as mticker


def plot_rollout_graph(
    R,
    feature_names=None,
    top_percent=5,
    top_k_out=None,
    min_edge_weight=None,
    figsize=(12, 6),
    cmap='viridis',
    seed=7,
    save_path=None,
    normalize=False,
    dpi=600
):
    """
    Plot global attention-flow graph from rollout matrix R.

    R[i, j] = attention flowing from feature i to feature j.
    Node size = received attention = column sum.
    Node color = outgoing entropy.
    """

    R = np.asarray(R, dtype=float)
    n = R.shape[0]

    if feature_names is None:
        feature_names = [f"x{j+1}" for j in range(n)]

    # Remove self-loops
    W = R.copy()
    np.fill_diagonal(W, 0.0)

    # Node statistics
    received_attention = R.sum(axis=0)

    if normalize: 
        row_sums = R.sum(axis=1, keepdims=True)
        P = np.divide(R, row_sums, out=np.zeros_like(R), where=row_sums > 0)
    else: 
        P = R
        
    entropy = -np.sum(np.where(P > 0, P * np.log(P), 0.0), axis=1)
    entropy = entropy / np.log(n)

    # Edge selection
    mask = np.zeros_like(W, dtype=bool)

    if top_k_out is not None:
        for i in range(n):
            idx = np.argsort(W[i, :])[-top_k_out:]
            mask[i, idx] = W[i, idx] > 0
    else:
        positive_weights = W[W > 0]
        threshold = np.percentile(positive_weights, 100 - top_percent)
        mask = W >= threshold

    if min_edge_weight is not None:
        mask &= W >= min_edge_weight

    # Build graph
    G = nx.DiGraph()
    for j, name in enumerate(feature_names):
        G.add_node(j, label=name)

    for i in range(n):
        for j in range(n):
            if mask[i, j]:
                G.add_edge(i, j, weight=W[i, j])

    # Layout
    pos = nx.spring_layout(G, weight="weight", seed=seed, k=1 / np.sqrt(n))

    # Visual scaling
    node_sizes = 1500 + 5000 * (
        (received_attention - received_attention.min())
        / (received_attention.max() - received_attention.min() + 1e-12)
    )

    edge_weights = np.array([G[u][v]["weight"] for u, v in G.edges()])
    edge_widths = 0.5 + 4.0 * (
        (edge_weights - edge_weights.min())
        / (edge_weights.max() - edge_weights.min() + 1e-12)
    ) if len(edge_weights) > 0 else []

    plt.figure(figsize=figsize)

    nodes = nx.draw_networkx_nodes(G,
                                   pos,
                                   node_size=node_sizes,
                                   node_color=entropy,
                                   cmap=cmap,
                                   alpha=0.9
                                  )

    nx.draw_networkx_edges(G, 
                           pos,
                           width=edge_widths, 
                           alpha=0.35, 
                           arrows=True, 
                           arrowsize=10, 
                           connectionstyle="arc3,rad=0.08"
                          )

    labels = nx.draw_networkx_labels(G, 
                                     pos, 
                                     labels={j: feature_names[j] for j in range(n)}, 
                                     font_size=16, 
                                     font_color="black")

    # Add white background (halo effect)
    for text in labels.values():
        text.set_bbox(dict(
            facecolor="white",
            edgecolor="none",
            alpha=0.8,
            boxstyle="round,pad=0.15"
        ))
  
    # cbar = plt.colorbar(nodes,
    #                     orientation="horizontal",
    #                     fraction=0.05,   # thickness of the bar
    #                     pad=0.02         # distance from the plot
    #                    )

    # cbar.ax.invert_xaxis() 

    # cbar.ax.tick_params(labelsize=14)
    # cbar.set_label("Normalized outgoing entropy", fontsize=18)

    # plt.title("Global attention-flow graph from rollout")
    plt.axis("off")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    plt.show()

    return G

def plot_feature_ranking_comparison(
    R,
    feature_names,
    top_k=15,
    outgoing_mass_k=5,
    selected_by="importance",   # "importance", "mass", or "union"
    save_path=None,
    dpi=600,
    figsize=(7.2, 5.0),
    fontsize=8,
    suptitle_pos=[0, 0, 0, 0]
):
    """
    Publication-style feature ranking figure.

    Panels:
    A. Received attention: column-sum rollout score
    B. Outgoing entropy: normalized row entropy
    C. Outgoing concentration: top-k outgoing mass

    The same feature set is used across panels.
    """

    R = np.asarray(R, dtype=float)
    n = R.shape[0]

    # -------------------------
    # Metrics
    # -------------------------
    importance = R.sum(axis=0)
    # importance = importance / (importance.sum() + 1e-12)

    row_sums = R.sum(axis=1, keepdims=True)
    P = np.divide(R, row_sums, out=np.zeros_like(R), where=row_sums > 0)

    entropy = -np.sum(np.where(P > 0, P * np.log(P), 0.0), axis=1)
    entropy = entropy / np.log(n)

    outgoing_mass = np.sort(P, axis=1)[:, -outgoing_mass_k:].sum(axis=1)

    # -------------------------
    # Select common feature set
    # -------------------------
    if selected_by == "importance":
        selected_idx = np.argsort(importance)[-top_k:][::-1]

    elif selected_by == "mass":
        selected_idx = np.argsort(outgoing_mass)[-top_k:][::-1]

    elif selected_by == "union":
        idx_imp = set(np.argsort(importance)[-top_k:])
        idx_mass = set(np.argsort(outgoing_mass)[-top_k:])
        selected_idx = np.array(list(idx_imp | idx_mass))
        selected_idx = selected_idx[np.argsort(importance[selected_idx])[::-1]]

    else:
        raise ValueError("selected_by must be 'importance', 'mass', or 'union'.")

    # Order by received attention for readability
    selected_idx = selected_idx[np.argsort(importance[selected_idx])[::-1]]

    names = np.array([feature_names[i] for i in selected_idx])
    imp_vals = importance[selected_idx]
    ent_vals = entropy[selected_idx]
    mass_vals = outgoing_mass[selected_idx]

    # Reverse so highest received attention appears at top
    names = names[::-1]
    imp_vals = imp_vals[::-1]
    ent_vals = ent_vals[::-1]
    mass_vals = mass_vals[::-1]

    y = np.arange(len(names))

    # -------------------------
    # Style
    # -------------------------
    plt.rcParams.update({
        "font.size": fontsize,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })

    fig, axes = plt.subplots(
        1, 3,
        figsize=figsize,
        sharey=True,
        gridspec_kw={
            "width_ratios": [1.25, 1.0, 1.0],
            "wspace": 0.12
        }
    )

    panels = [
        (imp_vals, "Received attention", "Column-sum score", "%.3f", "#007FFF"),
        (ent_vals, "Outgoing entropy", "Normalized entropy", "%.2f", "#FF55A3"),
        (mass_vals, "Outgoing concentration", f"Top-{outgoing_mass_k} mass", "%.2f", "#4CBB17"),
    ]

    for ax, (vals, title, xlabel, fmt, color) in zip(axes, panels):

        # Lollipop style
        ax.hlines(
            y=y,
            xmin=0,
            xmax=vals,
            color=color,
            alpha=0.35,
            linewidth=2.5
        )

        ax.scatter(
            vals,
            y,
            s=100,
            color=color,
            edgecolor="white",
            linewidth=2.5,
            zorder=3
        )

        # ax.set_title(title, fontsize=fontsize + 1, pad=6)
        ax.set_title(title, fontsize=10, pad=6)
        ax.set_xlabel(xlabel, fontsize=fontsize)

        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter(fmt))
        ax.tick_params(axis="x", labelsize=fontsize - 1)
        ax.tick_params(axis="y", length=0)

        ax.grid(axis="x", color="0.90", linewidth=0.5)
        ax.set_axisbelow(True)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_linewidth(0.6)

        ax.margins(x=0.08)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names, fontsize=fontsize)

    for ax in axes[1:]:
        ax.tick_params(labelleft=False)

    fig.text(
        suptitle_pos[0],
        suptitle_pos[1],
        "Feature roles in the attention rollout graph",
        ha="left",
        va="top",
        fontsize=fontsize + 2,
        fontweight="bold"
    )

    fig.text(
        suptitle_pos[2],
        suptitle_pos[3],
        f"Top {top_k} features selected by received attention. "
        f"Outgoing concentration is the top-{outgoing_mass_k} row mass.",
        ha="left",
        va="top",
        fontsize=fontsize,
        color="0.30"
    )

    # plt.tight_layout(rect=[0, 0, 1, 0.90])

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

        if save_path.endswith(".pdf"):
            plt.savefig(save_path.replace(".pdf", ".svg"), bbox_inches="tight")
        elif save_path.endswith(".png"):
            plt.savefig(save_path.replace(".png", ".pdf"), dpi=dpi, bbox_inches="tight")

    plt.show()

    return {
        "importance": importance,
        "entropy": entropy,
        "outgoing_mass": outgoing_mass,
        "selected_indices": selected_idx,
    }