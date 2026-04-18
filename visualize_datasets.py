"""
visualize_datasets.py

Standalone script to visualize sample graphs from the PROTEINS and ENZYMES
TUDataset benchmarks used in the DHN paper.

Saves two figures:
    - proteins_samples.png  (2 classes x 3 samples = 6 graphs)
    - enzymes_samples.png   (6 classes x 3 samples = 18 graphs)

Usage:
    python visualize_datasets.py
    python visualize_datasets.py --data_root ./data --samples_per_class 3 --dpi 150
"""

import argparse
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx

# ─── Helpers ──────────────────────────────────────────────────────────────────


def collect_samples(dataset, samples_per_class: int, seed: int = 42):
    """
    Return a dict {class_label: [Data, ...]} with `samples_per_class`
    examples per class, sampled reproducibly.
    """
    random.seed(seed)
    by_class = defaultdict(list)
    for i in range(len(dataset)):
        data = dataset[i]
        label = int(data.y.item())
        by_class[label].append(data)

    sampled = {}
    for label, items in sorted(by_class.items()):
        sampled[label] = random.sample(items, min(samples_per_class, len(items)))
    return sampled


def to_clean_nx_graph(data):
    """
    Convert PyG Data -> undirected NetworkX graph and remove self-loops.
    """
    G = to_networkx(data, to_undirected=True)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


def draw_graph(ax, data, title: str):
    """
    Draw a single PyG Data object on a matplotlib Axes.
    Nodes are coloured by degree (darker = higher degree).
    """
    G = to_clean_nx_graph(data)

    # Layout — kamada_kawai is more stable for biological graphs
    try:
        pos = nx.kamada_kawai_layout(G)
    except Exception:
        pos = nx.spring_layout(G, seed=42)

    degrees = np.array([d for _, d in G.degree()])
    if len(degrees) == 0 or degrees.max() == degrees.min():
        norm_degrees = np.zeros_like(degrees, dtype=float)
    else:
        norm_degrees = (degrees - degrees.min()) / (degrees.max() - degrees.min())

    # Map to a blue colormap — low degree = light, high degree = dark
    cmap = cm.get_cmap("Blues")
    node_colors = [cmap(0.3 + 0.65 * nd) for nd in norm_degrees] if len(degrees) else []

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#cccccc", width=0.8, alpha=0.7)
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_color=node_colors if len(node_colors) else "#8db3e2",
        node_size=60,
        linewidths=0.5,
        edgecolors="#555555",
    )

    ax.set_title(title, fontsize=8, pad=4)
    ax.axis("off")


def make_figure(
    dataset_name: str,
    sampled: dict,
    samples_per_class: int,
    class_names: dict,
    out_path: Path,
    dpi: int,
):
    """
    Build and save a grid figure: rows = classes, cols = samples.
    """
    n_classes = len(sampled)
    n_cols = samples_per_class
    n_rows = n_classes

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.2, n_rows * 3.0))

    # Ensure axes is always 2D
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for row_idx, (label, samples) in enumerate(sampled.items()):
        class_label = class_names.get(label, f"Class {label}")
        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]
            if col_idx < len(samples):
                data = samples[col_idx]
                G = to_clean_nx_graph(data)
                n_nodes = data.num_nodes
                n_edges = G.number_of_edges()  # robust undirected edge count
                title = f"{class_label}\nnodes={n_nodes}, edges={n_edges}"
                draw_graph(ax, data, title)
            else:
                ax.axis("off")

    # Colorbar legend for node degree
    sm = plt.cm.ScalarMappable(
        cmap=cm.get_cmap("Blues"), norm=mcolors.Normalize(vmin=0, vmax=1)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.4, aspect=20, pad=0.02)
    cbar.set_label("Node degree (normalised)", fontsize=9)
    cbar.set_ticks([0, 0.5, 1.0])
    cbar.set_ticklabels(["low", "mid", "high"])

    fig.suptitle(
        f"{dataset_name} — {samples_per_class} samples per class\n"
        f"Node colour = normalised degree",
        fontsize=11,
        y=1.01,
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ─── Dataset-specific class name maps ─────────────────────────────────────────

PROTEINS_CLASSES = {
    0: "Non-enzyme",
    1: "Enzyme",
}

# Enzyme Commission top-level categories
ENZYMES_CLASSES = {
    0: "Oxidoreductases",
    1: "Transferases",
    2: "Hydrolases",
    3: "Lyases",
    4: "Isomerases",
    5: "Ligases",
}


# ─── Main ─────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="Visualise DHN benchmark datasets")
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data",
        help="Root directory where datasets are stored",
    )
    parser.add_argument(
        "--samples_per_class",
        type=int,
        default=3,
        help="Number of sample graphs to show per class",
    )
    parser.add_argument("--dpi", type=int, default=150, help="Output image DPI")
    parser.add_argument(
        "--out_dir", type=str, default=".", help="Directory to save output images"
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        ("PROTEINS", PROTEINS_CLASSES, "proteins_samples.png"),
        ("ENZYMES", ENZYMES_CLASSES, "enzymes_samples.png"),
    ]

    for dataset_name, class_names, filename in configs:
        print(f"\nLoading {dataset_name}...")
        dataset = TUDataset(
            root=args.data_root,
            name=dataset_name,
            use_node_attr=True,
        )
        print(
            f"  {len(dataset)} graphs, "
            f"{dataset.num_classes} classes, "
            f"{dataset.num_node_features} node features"
        )

        sampled = collect_samples(dataset, args.samples_per_class, seed=args.seed)

        make_figure(
            dataset_name=dataset_name,
            sampled=sampled,
            samples_per_class=args.samples_per_class,
            class_names=class_names,
            out_path=out_dir / filename,
            dpi=args.dpi,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
