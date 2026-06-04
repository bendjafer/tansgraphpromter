"""
Shared preprocessing for Brazil / Europe / USA airport datasets.

Reads:
    dataset/airports/{name}/topo.npy         (5 topo features, cached by generate_descriptions.py)
    dataset/airports/{name}/descriptions.csv (LLM descriptions, cached by generate_descriptions.py)

Writes:
    dataset/tape_{name}/processed/data.pt
    dataset/tape_{name}/processed/text.csv
    dataset/tape_{name}/split/{train,val,test}_indices.txt

Run for a single dataset:
    python -m src.dataset.preprocess.airports --dataset brazil
    python -m src.dataset.preprocess.airports --dataset europe
"""

import argparse
import os
import numpy as np
from torch_geometric.utils import to_undirected
import pandas as pd
import torch
from torch_geometric.datasets import Airports
from torch_geometric.utils import to_networkx
import networkx as nx

from src.dataset.preprocess.generate_split import generate_split


AIRPORT_ROOT  = "dataset/airports"
CLASSES       = ["Low", "Medium", "High", "Very High"]
FEATURE_ORDER = ["degree", "closeness", "betweenness", "clustering", "square_clustering"]

DATASET_META = {
    "brazil": {"num_nodes": 131,  "pyg_name": "Brazil"},
    "europe": {"num_nodes": 399,  "pyg_name": "Europe"},
    "usa":    {"num_nodes": 1190, "pyg_name": "USA"},
}


def _topo_path(name: str) -> str:
    return f"{AIRPORT_ROOT}/{name}/topo.npy"

def _desc_path(name: str) -> str:
    return f"{AIRPORT_ROOT}/{name}/descriptions.csv"


def compute_and_save_topo(data, name: str) -> dict:
    """Compute 5 topo features via NetworkX and cache to disk."""
    print(f"  Computing topology for {name} ...")
    G = to_networkx(data, to_undirected=True)
    topo = {
        "degree":            nx.degree_centrality(G),
        "closeness":         nx.closeness_centrality(G),
        "betweenness":       nx.betweenness_centrality(G, normalized=True),
        "clustering":        nx.clustering(G),
        "square_clustering": nx.square_clustering(G),
    }
    path = _topo_path(name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, topo)
    print(f"  Saved topo -> {path}")
    return topo


def load_topo(name: str, data) -> dict:
    """Load topo from cache, compute and save if missing."""
    path = _topo_path(name)
    if os.path.exists(path):
        print(f"  Loading topo from {path}")
        return np.load(path, allow_pickle=True).item()
    return compute_and_save_topo(data, name)


def build_feature_matrix(topo: dict, num_nodes: int) -> np.ndarray:
    """Stack the 5 topo dicts into a [num_nodes, 5] float32 array."""
    return np.array(
        [[topo[k][i] for k in FEATURE_ORDER] for i in range(num_nodes)],
        dtype=np.float32,
    )


def preprocess(name: str):
    meta      = DATASET_META[name]
    num_nodes = meta["num_nodes"]
    pyg_name  = meta["pyg_name"]

    print(f"\nPreprocessing {name} airport dataset ...")

    # ── Graph ───────────────────────────────────────────────────────────────
    dataset = Airports(root=AIRPORT_ROOT, name=pyg_name)
    data    = dataset[0]

    # ── Topo features → graph.x ─────────────────────────────────────────────
    topo   = load_topo(name, data)
    x      = build_feature_matrix(topo, num_nodes)
    data.x = torch.tensor(x, dtype=torch.float)   # [N, 5]

    # Make edge_index undirected: 93 % of airport edges are one-way in the raw data,
    # which would starve many nodes of incoming messages and is inconsistent with the
    # topo features (which were computed on the undirected graph).
    data.edge_index = to_undirected(data.edge_index, num_nodes=num_nodes)

    # ── Labels ──────────────────────────────────────────────────────────────
    labels = [CLASSES[y.item()] for y in data.y]

    # ── Descriptions ────────────────────────────────────────────────────────
    desc_path = _desc_path(name)
    if not os.path.exists(desc_path):
        raise FileNotFoundError(
            f"\n{desc_path} not found.\n"
            f"Generate it first:\n"
            f"  OPENAI_API_KEY=sk-...  python generate_descriptions.py --dataset {name}\n"
        )
    desc_df      = pd.read_csv(desc_path).sort_values("node_idx").reset_index(drop=True)
    if len(desc_df) != num_nodes:
        raise ValueError(f"Expected {num_nodes} descriptions, got {len(desc_df)}")
    descriptions = desc_df["description"].tolist()

    # ── Save ────────────────────────────────────────────────────────────────
    out_dir = f"dataset/tape_{name}/processed"
    os.makedirs(out_dir, exist_ok=True)

    torch.save(data, f"{out_dir}/data.pt")
    print(f"  Saved graph  -> {out_dir}/data.pt  (x: {data.x.shape})")

    df = pd.DataFrame({
        "node_id":     np.arange(num_nodes),
        "label":       labels,
        "description": descriptions,
    })
    df.to_csv(f"{out_dir}/text.csv", index=False)
    print(f"  Saved text   -> {out_dir}/text.csv")

    generate_split(num_nodes, f"dataset/tape_{name}/split")
    print(f"  Saved split  -> dataset/tape_{name}/split/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["brazil", "europe", "usa"])
    args = parser.parse_args()
    preprocess(args.dataset)
    print("Done!")
