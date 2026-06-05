"""
Feature separation analysis for airport graph datasets.

For each combination of (dataset: brazil|europe) × (features: topo|minilm):
  Step 1 — Visualise: t-SNE and PCA projections coloured by activity class
  Step 2 — Quantify: silhouette, Davies-Bouldin, k-NN probe, linear probe
  Step 3 — Summary table

Run:
    python feature_analysis.py
Plots saved to:  analysis_plots/
"""

import os
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

os.makedirs('analysis_plots', exist_ok=True)

# ── Class ordering and colour palette ─────────────────────────────────────────
CLASS_ORDER  = ['Low', 'Medium', 'High', 'Very High']
CLASS_COLORS = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2']

LABEL_MAP = {c: i for i, c in enumerate(CLASS_ORDER)}

# ── Data loading ──────────────────────────────────────────────────────────────

FEATURE_ORDER = ['degree', 'closeness', 'betweenness', 'clustering', 'square_clustering']


def load_data(name: str):
    """Return (topo [N,5], minilm [N,384], labels [N] int) for brazil or europe."""
    text_csv  = f'dataset/tape_{name}/processed/text.csv'
    topo_npy  = f'dataset/airports/{name}/topo.npy'
    minilm_pt = f'dataset/airports/{name}/minilm_x.pt'

    df = pd.read_csv(text_csv).sort_values('node_id').reset_index(drop=True)
    N  = len(df)

    # topo.npy is a 0-d object array wrapping a dict-of-dicts {feat: {node_id: value}}
    topo_dict = np.load(topo_npy, allow_pickle=True).item()
    topo = np.array(
        [[topo_dict[k][i] for k in FEATURE_ORDER] for i in range(N)],
        dtype=np.float32,
    )  # [N, 5]

    minilm = torch.load(minilm_pt, weights_only=True).numpy()  # [N, 384]
    labels = np.array([LABEL_MAP[l] for l in df['label']])

    assert topo.shape[0] == minilm.shape[0] == len(labels), \
        f"{name}: shape mismatch topo={topo.shape} minilm={minilm.shape} labels={len(labels)}"

    return topo, minilm, labels


# ── Step 1 — Visualisation ───────────────────────────────────────────────────

def make_legend(ax):
    patches = [mpatches.Patch(color=CLASS_COLORS[i], label=CLASS_ORDER[i])
               for i in range(len(CLASS_ORDER))]
    ax.legend(handles=patches, fontsize=8, loc='best', framealpha=0.7)


def plot_projections(X_scaled, labels, dataset_name, feat_name):
    """PCA (left) and t-SNE (right) side-by-side, saved to analysis_plots/."""
    colors = [CLASS_COLORS[l] for l in labels]

    # PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    var_explained = pca.explained_variance_ratio_.sum() * 100

    # t-SNE
    perplexity = min(30, len(labels) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity,
                n_iter=1000, init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(X_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'{dataset_name.capitalize()} — {feat_name} features', fontsize=13, fontweight='bold')

    for ax, X_2d, title in [
        (axes[0], X_pca,  f'PCA  ({var_explained:.1f}% var)'),
        (axes[1], X_tsne, 't-SNE'),
    ]:
        ax.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, s=40, alpha=0.8, linewidths=0)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        make_legend(ax)

    plt.tight_layout()
    path = f'analysis_plots/{dataset_name}_{feat_name}.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


# ── Step 2 — Quantitative probes ────────────────────────────────────────────

def quantify(X_scaled, labels):
    """Return dict with silhouette, davies_bouldin, knn_acc, linear_acc."""
    sil   = silhouette_score(X_scaled, labels)
    db    = davies_bouldin_score(X_scaled, labels)

    knn   = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    knn_acc = cross_val_score(knn, X_scaled, labels, cv=5, scoring='accuracy').mean()

    lr    = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    lr_acc = cross_val_score(lr, X_scaled, labels, cv=5, scoring='accuracy').mean()

    return {'silhouette': sil, 'davies_bouldin': db,
            'knn_acc': knn_acc, 'linear_acc': lr_acc}


# ── Main ─────────────────────────────────────────────────────────────────────

DATASETS  = ['brazil', 'europe']
FEAT_SETS = {'topo': 0, 'minilm': 1}  # index into load_data return tuple

results = []

for dataset_name in DATASETS:
    print(f'\n{"="*60}')
    print(f'  Dataset: {dataset_name.upper()}')
    print(f'{"="*60}')

    topo, minilm, labels = load_data(dataset_name)

    for feat_name, feat_arr in [('topo', topo), ('minilm', minilm)]:
        print(f'\n  ── {feat_name} features  {feat_arr.shape} ──')

        # Standardise
        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(feat_arr)

        # ── Step 1 ──
        print('  Step 1: projecting to 2D …', end=' ', flush=True)
        plot_projections(X_scaled, labels, dataset_name, feat_name)

        # ── Step 2 ──
        print('  Step 2: computing separation metrics …', end=' ', flush=True)
        metrics = quantify(X_scaled, labels)
        print('done')
        for k, v in metrics.items():
            print(f'    {k:20s}: {v:.4f}')

        results.append({'dataset': dataset_name, 'features': feat_name, **metrics})

# ── Step 3 — Summary table ───────────────────────────────────────────────────
print(f'\n{"="*60}')
print('  SUMMARY TABLE')
print(f'{"="*60}')

df = pd.DataFrame(results)
df = df.set_index(['dataset', 'features'])
df.columns = ['Silhouette ↑', 'Davies-Bouldin ↓', 'kNN acc ↑', 'Linear acc ↑']
df = df.round(4)

# Console display
print(df.to_string())

# Save as CSV
df.to_csv('analysis_plots/summary.csv')
print('\nSaved: analysis_plots/summary.csv')

# ── Side-by-side comparison plot: topo vs minilm per dataset ─────────────────
print('\nGenerating comparison grid …', end=' ', flush=True)

fig, axes = plt.subplots(2, 4, figsize=(22, 10))
fig.suptitle('Topo vs MiniLM — PCA and t-SNE by dataset', fontsize=13, fontweight='bold')

col_titles = ['PCA (topo)', 't-SNE (topo)', 'PCA (MiniLM)', 't-SNE (MiniLM)']
for ax, title in zip(axes[0], col_titles):
    ax.set_title(title, fontsize=10, fontweight='bold')

for row_i, dataset_name in enumerate(DATASETS):
    topo, minilm, labels = load_data(dataset_name)
    colors = [CLASS_COLORS[l] for l in labels]

    for feat_name, feat_arr in [('topo', topo), ('minilm', minilm)]:
        col_offset = 0 if feat_name == 'topo' else 2
        X_scaled = StandardScaler().fit_transform(feat_arr)

        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)

        perplexity = min(30, len(labels) - 1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity,
                    n_iter=1000, init='pca', learning_rate='auto')
        X_tsne = tsne.fit_transform(X_scaled)

        for col_j, X_2d in enumerate([X_pca, X_tsne]):
            ax = axes[row_i][col_offset + col_j]
            ax.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, s=30, alpha=0.8, linewidths=0)
            ax.set_ylabel(dataset_name.capitalize(), fontsize=10, fontweight='bold')
            ax.set_xticks([]); ax.set_yticks([])
            if row_i == 0 and col_j == 0 and feat_name == 'topo':
                make_legend(ax)

plt.tight_layout()
grid_path = 'analysis_plots/comparison_grid.png'
plt.savefig(grid_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'done\nSaved: {grid_path}')
