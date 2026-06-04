"""
General workflow to build airport node structural descriptions.

STEP 1 — Compute & save topology
    Load the PyG Airports graph for the chosen dataset, compute the 5 TANS
    topological properties with NetworkX, and save to
    dataset/airports/{name}/topo.npy.
    Skipped automatically if the file already exists.

STEP 2 — Generate descriptions via OpenAI API
    Load the saved topology, call the API once per node using a few-shot
    prompt (same neutral-factual style as the original TANS script), and
    write results to dataset/airports/{name}/descriptions.csv.

Usage:
    python generate_descriptions.py --dataset brazil
    python generate_descriptions.py --dataset europe
    python generate_descriptions.py --dataset usa
    python generate_descriptions.py --dataset europe --force-recompute
    python generate_descriptions.py --dataset brazil --model-name gpt-4o

The OpenAI API key is loaded automatically from the .env file in the project root.
"""

import argparse
import csv
import os
import time
from pathlib import Path

from dotenv import load_dotenv
import networkx as nx
import numpy as np
from openai import OpenAI, OpenAIError
from torch_geometric.datasets import Airports

# Load OPENAI_API_KEY from .env if present (does nothing if already set in env)
load_dotenv(Path(__file__).resolve().parent / ".env")

# ---------------------------------------------------------------------------
# Per-dataset metadata
# ---------------------------------------------------------------------------
DATASET_META = {
    "brazil": {"num_nodes": 131,  "num_edges": 1074},
    "europe": {"num_nodes": 399,  "num_edges": 5995},
    "usa":    {"num_nodes": 1190, "num_edges": 13599},
}

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
PYG_ROOT = Path(__file__).resolve().parent / "dataset" / "airports"


def _paths(dataset_name: str):
    base = PYG_ROOT / dataset_name
    return base / "topo.npy", base / "descriptions.csv"


# ===========================================================================
# STEP 1 — Topology
# ===========================================================================

def compute_topology(dataset_name: str) -> dict:
    data = Airports(root=str(PYG_ROOT), name=dataset_name.capitalize())[0]
    num_nodes = DATASET_META[dataset_name]["num_nodes"]

    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    for u, v in zip(*data.edge_index.tolist()):
        G.add_edge(u, v)

    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} undirected edges")
    print("  Computing degree centrality ...")
    print("  Computing closeness centrality ...")
    print("  Computing betweenness centrality ...")
    print("  Computing clustering coefficient ...")
    print("  Computing square clustering coefficient ...")
    return {
        "degree":            nx.degree_centrality(G),
        "closeness":         nx.closeness_centrality(G),
        "betweenness":       nx.betweenness_centrality(G, normalized=True),
        "clustering":        nx.clustering(G),
        "square_clustering": nx.square_clustering(G),
    }


def run_step1(dataset_name: str, force: bool) -> dict:
    topo_file, _ = _paths(dataset_name)
    num_nodes = DATASET_META[dataset_name]["num_nodes"]

    print("\n" + "=" * 60)
    print(f"STEP 1 — Topology computation  [{dataset_name}]")
    print("=" * 60)

    if topo_file.exists() and not force:
        print(f"  Found existing file: {topo_file}")
        print("  Loading ... (use --force-recompute to recompute)")
        topo = np.load(topo_file, allow_pickle=True).item()
    else:
        topo = compute_topology(dataset_name)
        topo_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(topo_file, topo)
        print(f"  Saved -> {topo_file}")

    for key in ("degree", "closeness", "betweenness", "clustering", "square_clustering"):
        assert key in topo and len(topo[key]) == num_nodes, f"Bad topo key: {key}"
    print(f"  OK — 5 properties x {num_nodes} nodes")
    print(f"  Node-0: degree={topo['degree'][0]:.4f}  closeness={topo['closeness'][0]:.4f}  "
          f"betweenness={topo['betweenness'][0]:.4f}  clustering={topo['clustering'][0]:.4f}  "
          f"sq_clust={topo['square_clustering'][0]:.4f}")
    return topo


# ===========================================================================
# STEP 2 — Descriptions
# ===========================================================================

SYSTEM_PROMPT = (
    "You are an expert in graph theory and network topology. "
    "Your task is to write a precise, concrete structural description of a graph node "
    "based solely on its five topological metrics.\n\n"
    "Follow these rules:\n"
    "- For each metric, state its value AND derive a concrete structural fact from it. "
    "For example: convert normalized degree to an actual neighbor count, convert closeness "
    "to an average shortest-path length, convert betweenness to a percentage of paths, "
    "convert clustering to a percentage of mutually connected neighbor pairs.\n"
    "- Explain what each metric captures structurally in the network.\n"
    "- Use factual numerical statements (counts, percentages, averages derived from the "
    "given values and graph size). These are encouraged.\n"
    "- Do NOT interpret the values in terms of real-world meaning or role.\n"
    "- Do NOT use evaluative language ('important', 'strong', 'weak', 'peripheral', "
    "'dominant', 'influential', or similar).\n"
    "- Do NOT rank or compare the node to other nodes.\n"
    "- Write in plain, precise sentences. Under 200 words."
)

# One-shot example: Brazil node-0.
# Derived facts used in the answer:
#   degree 0.3077 × (131-1) = 40 direct neighbors
#   closeness 0.5856  →  avg path = 1/0.5856 ≈ 1.71 hops
#   betweenness 0.0083 → 0.83 % of all inter-node shortest paths
#   clustering 0.5960  → 59.6 % of neighbor pairs are mutually connected
#   square_clustering 0.3343 → 33.4 % rate of 4-cycle formation
_EXAMPLE = (
    "Here is an example:\n\n"
    "Given a node from an airport network graph with 131 nodes and 1074 edges.\n"
    "The node has the following structural properties:\n"
    "- Node Degree: 0.3077\n"
    "- Closeness Centrality: 0.5856\n"
    "- Betweenness Centrality: 0.0083\n"
    "- Clustering Coefficient: 0.5960\n"
    "- Square Clustering Coefficient: 0.3343\n\n"
    "Answer:\n"
    "This node has direct edges to 40 out of the 130 other nodes in the network "
    "(degree centrality 0.3077 × 130 ≈ 40). "
    "Its closeness centrality of 0.5856 corresponds to a mean shortest-path distance of "
    "1 / 0.5856 ≈ 1.71 hops to reach any other node in the graph. "
    "The betweenness centrality of 0.0083 means this node lies on approximately 0.83% of "
    "all shortest paths between every other pair of nodes. "
    "The clustering coefficient of 0.5960 indicates that about 59.6% of the node's 40 "
    "neighbors share a direct edge with one another, forming a dense local neighbourhood. "
    "The square clustering coefficient of 0.3343 measures the fraction of potential "
    "four-node cycles (squares) that are closed among the node's neighbors; here, 33.4% "
    "of such cycles are present, reflecting the extent to which neighbours are connected "
    "through paths of length two as well as direct edges.\n\n"
)


def build_prompt(node_idx: int, topo: dict, dataset_name: str) -> str:
    meta = DATASET_META[dataset_name]
    n, e = meta["num_nodes"], meta["num_edges"]
    d  = f"{topo['degree'][node_idx]:.4f}"
    cl = f"{topo['closeness'][node_idx]:.4f}"
    bw = f"{topo['betweenness'][node_idx]:.4f}"
    ct = f"{topo['clustering'][node_idx]:.4f}"
    sq = f"{topo['square_clustering'][node_idx]:.4f}"

    # Derive neighbor count so the LLM can reference it in the description
    neighbor_count = round(topo['degree'][node_idx] * (n - 1))

    return (
        _EXAMPLE +
        f"Now describe the following node in the same style, deriving concrete facts "
        f"(e.g., neighbor count = degree × {n - 1}, avg path length = 1 / closeness, "
        f"betweenness as a percentage, clustering and square-clustering as percentages).\n\n"
        f"Given a node from an airport network graph with {n} nodes and {e} edges.\n"
        f"The node has {neighbor_count} direct neighbours.\n"
        "The node has the following structural properties:\n"
        f"- Node Degree: {d}\n"
        f"- Closeness Centrality: {cl}\n"
        f"- Betweenness Centrality: {bw}\n"
        f"- Clustering Coefficient: {ct}\n"
        f"- Square Clustering Coefficient: {sq}\n\n"
        "Answer:\n"
    )


def query_llm(client: OpenAI, model_name: str, prompt: str, max_retries: int = 3) -> str:
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                max_completion_tokens=512,
            )
            return " ".join((resp.choices[0].message.content or "").strip().split())
        except OpenAIError as e:
            print(f"    [retry {attempt + 1}] {e}")
            time.sleep(2 * (attempt + 1))
    return "Error"


def run_step2(dataset_name: str, topo: dict, client: OpenAI, model_name: str) -> None:
    _, desc_file = _paths(dataset_name)
    num_nodes = DATASET_META[dataset_name]["num_nodes"]

    print("\n" + "=" * 60)
    print(f"STEP 2 — LLM description generation  [{dataset_name}]")
    print("=" * 60)
    print(f"  Model : {model_name}")
    print(f"  Nodes : {num_nodes}")
    print(f"  Output: {desc_file}\n")

    rows = []
    for node_idx in range(num_nodes):
        prompt = build_prompt(node_idx, topo, dataset_name)
        desc   = query_llm(client, model_name, prompt)
        rows.append({"node_idx": node_idx, "description": desc})
        print(f"  [{node_idx:3d}/{num_nodes}] {desc[:80]}...")

    desc_file.parent.mkdir(parents=True, exist_ok=True)
    with desc_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["node_idx", "description"])
        writer.writeheader()
        writer.writerows(rows)

    errors = sum(1 for r in rows if r["description"] == "Error")
    print(f"\n  Done — {num_nodes - errors}/{num_nodes} descriptions written.")
    if errors:
        print(f"  WARNING: {errors} nodes returned 'Error'. Re-run to retry.")
    print(f"  Saved -> {desc_file}")


# ===========================================================================
# Entry point
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute topo features and generate LLM descriptions for airport datasets."
    )
    parser.add_argument("--dataset", required=True, choices=["brazil", "europe", "usa"],
                        help="Airport dataset to process.")
    parser.add_argument("--force-recompute", action="store_true",
                        help="Recompute topology even if topo.npy already exists.")
    parser.add_argument("--model-name", default="gpt-4o-mini",
                        help="OpenAI chat model (default: gpt-4o-mini).")
    return parser.parse_args()


def main():
    if not OPENAI_API_KEY:
        raise EnvironmentError("Please set the OPENAI_API_KEY environment variable.")

    args   = parse_args()
    client = OpenAI(api_key=OPENAI_API_KEY)

    topo = run_step1(args.dataset, force=args.force_recompute)
    run_step2(args.dataset, topo, client, args.model_name)


if __name__ == "__main__":
    main()
