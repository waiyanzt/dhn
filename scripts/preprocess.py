"""
Preprocess raw EXP, SR25, and CSL data into .pkl format expected by HomDataset.

Expected pkl format: (num_classes, num_features, list_of_networkx_graphs)
Each graph g must have:
    g.graph['x'] : torch.Tensor of shape (num_nodes, num_features)
    g.graph['y'] : torch.Tensor of shape (1,) with class label

Usage:
    python scripts/preprocess.py --data_dir ./data
"""

import argparse
import os
import pickle

import networkx as nx
import torch

# ---------------------------------------------------------------------------
# EXP
# ---------------------------------------------------------------------------


def parse_exp(path):
    """
    EXP.txt format:
        First line: total number of graphs
        Per graph:
            <num_nodes> <label>
            <node_label> <num_neighbours> <neighbour_ids...>   (one per node)
    Node features: one-hot encoding of node label (0 or 1) -> shape (n, 2)
    """
    graphs = []
    with open(path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    idx = 0
    num_graphs = int(lines[idx])
    idx += 1

    for _ in range(num_graphs):
        header = lines[idx].split()
        idx += 1
        num_nodes = int(header[0])
        label = int(header[1])

        g = nx.Graph()
        g.add_nodes_from(range(num_nodes))
        node_labels = []

        for node_id in range(num_nodes):
            parts = lines[idx].split()
            idx += 1
            node_label = int(parts[0])
            num_neighbors = int(parts[1])
            neighbors = [int(parts[2 + k]) for k in range(num_neighbors)]
            node_labels.append(node_label)
            for nb in neighbors:
                if nb > node_id:  # avoid duplicate edges
                    g.add_edge(node_id, nb)

        # One-hot node features from binary label
        x = torch.zeros(num_nodes, 2)
        for i, nl in enumerate(node_labels):
            x[i, nl] = 1.0

        g.graph["x"] = x
        g.graph["y"] = torch.tensor([label], dtype=torch.long)
        graphs.append(g)

    num_classes = len(set(g.graph["y"].item() for g in graphs))
    num_features = 2
    return num_classes, num_features, graphs


# ---------------------------------------------------------------------------
# SR25
# ---------------------------------------------------------------------------


def parse_sr25(path):
    """
    sr25.g6: graph6 format, 15 strongly regular graphs on 25 nodes.
    All graphs are in the same isomorphism class for 1-WL/3-WL,
    so the task is distinguishing them -> treat as single class (15 graphs, 1 class).
    Node features: constant ones, shape (25, 1).
    Labels: graph index mod num_classes. Following the paper which uses
    a multiclass setup where each graph is its own class (15 classes).
    """
    raw_graphs = nx.read_graph6(path)
    # read_graph6 returns a single graph or list depending on version
    if isinstance(raw_graphs, nx.Graph):
        raw_graphs = [raw_graphs]

    graphs = []
    for i, rg in enumerate(raw_graphs):
        g = nx.Graph(rg)
        num_nodes = g.number_of_nodes()
        g.graph["x"] = torch.ones(num_nodes, 1)
        g.graph["y"] = torch.tensor([i], dtype=torch.long)
        graphs.append(g)

    num_classes = len(graphs)
    num_features = 1
    return num_classes, num_features, graphs


# ---------------------------------------------------------------------------
# CSL (Circular Skip Links)
# ---------------------------------------------------------------------------


def generate_csl():
    """
    CSL: 150 4-regular graphs on 41 nodes.
    Constructed as circulant graphs C(41, {1, s}) for s in skip_links,
    repeated 10 times each (150 total, 10 classes of 15 graphs).
    Skip values from the original paper: {2,3,4,5,6,7,8,9,10,11} -> 10 classes.
    Node features: constant ones, shape (41, 1).
    Label: index of skip value (0-9).
    """
    skip_links = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    n = 41
    num_per_class = 15
    graphs = []

    for label, s in enumerate(skip_links):
        for _ in range(num_per_class):
            g = nx.circulant_graph(n, [1, s])
            g.graph["x"] = torch.ones(n, 1)
            g.graph["y"] = torch.tensor([label], dtype=torch.long)
            graphs.append(g)

    num_classes = len(skip_links)
    num_features = 1
    return num_classes, num_features, graphs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def save_pkl(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Directory containing raw data files and where pkl will be saved",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    os.makedirs(data_dir, exist_ok=True)

    # EXP
    exp_txt = os.path.join(data_dir, "EXP.txt")
    exp_pkl = os.path.join(data_dir, "exp.pkl")
    if os.path.exists(exp_txt):
        print("Processing EXP...")
        data = parse_exp(exp_txt)
        save_pkl(data, exp_pkl)
    else:
        print(f"WARNING: {exp_txt} not found, skipping EXP.")

    # SR25
    sr25_g6 = os.path.join(data_dir, "sr25.g6")
    sr25_pkl = os.path.join(data_dir, "sr25.pkl")
    if os.path.exists(sr25_g6):
        print("Processing SR25...")
        data = parse_sr25(sr25_g6)
        save_pkl(data, sr25_pkl)
    else:
        print(f"WARNING: {sr25_g6} not found, skipping SR25.")

    # CSL (generated programmatically, no raw file needed)
    csl_pkl = os.path.join(data_dir, "csl.pkl")
    print("Generating CSL...")
    data = generate_csl()
    save_pkl(data, csl_pkl)

    print("\nDone. Files written:")
    for name in ["exp.pkl", "sr25.pkl", "csl.pkl"]:
        p = os.path.join(data_dir, name)
        status = "OK" if os.path.exists(p) else "MISSING"
        print(f"  {p}: {status}")


if __name__ == "__main__":
    main()
