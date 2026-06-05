"""
Shared preprocessing for Brazil / Europe / USA airport datasets.

Reads:
    dataset/airports/{name}/topo.npy         (5 topo features)
    dataset/airports/{name}/descriptions.csv (LLM descriptions)

Writes (--features topo, default):
    dataset/tape_{name}/processed/data.pt     x = [N, 5] topo matrix
    dataset/tape_{name}/processed/text.csv
    dataset/tape_{name}/split/

Writes (--features minilm):
    dataset/airports/{name}/minilm_x.pt       cached MiniLM embeddings [N, 384]
    dataset/tape_{name}_minilm/processed/data.pt  x = [N, 384]
    dataset/tape_{name}_minilm/processed/text.csv
    dataset/tape_{name}_minilm/split/         reuses same split as topo variant

Run:
    python -m src.dataset.preprocess.airports --dataset brazil
    python -m src.dataset.preprocess.airports --dataset europe --features minilm
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


# ── MiniLM encoding ─────────────────────────────────────────────────────────

MINILM_MODEL   = "all-MiniLM-L12-v2"   # same model as encode/encode_text.py
# Same template as encode_text.py line 99; for airports default_text is None → 'na'
_DESC_TEMPLATE = (
    "The node type is airport. "
    "The node description is na. "
    "The additional node description is {}."
)


def _minilm_cache_path(name: str) -> str:
    return f"{AIRPORT_ROOT}/{name}/minilm_x.pt"


def encode_and_save_minilm(descriptions: list, name: str) -> torch.Tensor:
    """Encode descriptions with MiniLM-L12-v2 and cache to disk."""
    from sentence_transformers import SentenceTransformer
    import tqdm

    print(f"  Encoding {len(descriptions)} descriptions with {MINILM_MODEL} ...")
    model = SentenceTransformer(MINILM_MODEL)

    embeddings = []
    for desc in tqdm.tqdm(descriptions, desc="  MiniLM"):
        text = _DESC_TEMPLATE.format(desc)
        embeddings.append(model.encode(text))           # same as encode_text_sbert()

    x = torch.tensor(embeddings, dtype=torch.float32)  # [N, 384]
    cache = _minilm_cache_path(name)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    torch.save(x, cache)
    print(f"  Saved MiniLM embeddings -> {cache}  (shape: {x.shape})")
    return x


def load_minilm_embeddings(descriptions: list, name: str) -> torch.Tensor:
    """Load MiniLM embeddings from cache, encode and save if missing."""
    cache = _minilm_cache_path(name)
    if os.path.exists(cache):
        print(f"  Loading MiniLM embeddings from {cache}")
        x = torch.load(cache, weights_only=True)
        print(f"  Shape: {x.shape}")
        return x
    return encode_and_save_minilm(descriptions, name)


# ── Main preprocess entry point ──────────────────────────────────────────────

def preprocess(name: str, features: str = "topo"):
    assert features in ("topo", "minilm"), f"Unknown features: {features}"
    meta      = DATASET_META[name]
    num_nodes = meta["num_nodes"]
    pyg_name  = meta["pyg_name"]
    tag       = name if features == "topo" else f"{name}_minilm"

    print(f"\nPreprocessing {tag} airport dataset (features={features}) ...")

    # ── Graph ───────────────────────────────────────────────────────────────
    dataset = Airports(root=AIRPORT_ROOT, name=pyg_name)
    data    = dataset[0]

    # Make edge_index undirected (93 % of raw edges are one-way)
    data.edge_index = to_undirected(data.edge_index, num_nodes=num_nodes)

    # ── Labels ──────────────────────────────────────────────────────────────
    labels = [CLASSES[y.item()] for y in data.y]

    # ── Descriptions (needed for both feature types) ─────────────────────────
    desc_path = _desc_path(name)
    if not os.path.exists(desc_path):
        raise FileNotFoundError(
            f"\n{desc_path} not found.\n"
            f"Generate it first:\n"
            f"  python generate_descriptions.py --dataset {name}\n"
        )
    desc_df      = pd.read_csv(desc_path).sort_values("node_idx").reset_index(drop=True)
    if len(desc_df) != num_nodes:
        raise ValueError(f"Expected {num_nodes} descriptions, got {len(desc_df)}")
    descriptions = desc_df["description"].tolist()

    # ── Feature matrix → graph.x ─────────────────────────────────────────────
    if features == "topo":
        topo   = load_topo(name, data)
        x      = build_feature_matrix(topo, num_nodes)
        data.x = torch.tensor(x, dtype=torch.float)          # [N, 5]
    else:
        data.x = load_minilm_embeddings(descriptions, name)  # [N, 384]

    # ── Save ────────────────────────────────────────────────────────────────
    out_dir   = f"dataset/tape_{tag}/processed"
    split_dir = f"dataset/tape_{tag}/split"
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

    generate_split(num_nodes, split_dir)
    print(f"  Saved split  -> {split_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",  required=True, choices=["brazil", "europe", "usa"])
    parser.add_argument("--features", default="topo", choices=["topo", "minilm"],
                        help="topo: 5-dim topological matrix (default); minilm: 384-dim MiniLM embeddings")
    args = parser.parse_args()
    preprocess(args.dataset, features=args.features)
    print("Done!")
