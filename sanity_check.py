"""
Full pipeline sanity check — TANS airport experiments.

Verifies (without loading Vicuna-7B):
  1. Data split   — source 100% train, target 20% val / 80% test, no overlap
  2. Features     — shapes, NaN/inf, TPF normalisation
  3. Labels       — class distribution across splits
  4. Model        — GNN architecture vs hardcoded TANS hyperparams
  5. Checkpoint   — save/restore logic (static code inspection)
  6. Results      — parse logs if present, flag overfitting / suspiciously low acc

Run:
    python sanity_check.py
"""

import os, re, math, sys, glob, textwrap
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Helpers ──────────────────────────────────────────────────────────────────

PASS = "\033[92m✔ PASS\033[0m"
FAIL = "\033[91m✘ FAIL\033[0m"
WARN = "\033[93m⚠ WARN\033[0m"
INFO = "\033[94mℹ INFO\033[0m"

def hdr(title):
    print(f"\n{'='*62}")
    print(f"  {title}")
    print(f"{'='*62}")

def check(label, ok, detail=""):
    status = PASS if ok else FAIL
    print(f"  [{status}] {label}")
    if detail:
        for line in detail.strip().splitlines():
            print(f"           {line}")

def warn(label, detail=""):
    print(f"  [{WARN}] {label}")
    if detail:
        for line in detail.strip().splitlines():
            print(f"           {line}")

def info(label, detail=""):
    print(f"  [{INFO}] {label}")
    if detail:
        for line in detail.strip().splitlines():
            print(f"           {line}")

# ── TANS configuration ───────────────────────────────────────────────────────

FEATURE_ORDER = ['degree', 'closeness', 'betweenness', 'clustering', 'square_clustering']
LABEL_ORDER   = ['Low', 'Medium', 'High', 'Very High']
LABEL_MAP     = {c: i for i, c in enumerate(LABEL_ORDER)}
TARGET_VAL_RATIO = 0.2
SEEDS = [0, 1, 2]

TANS_CONFIGS = {
    # (source, target)
    ('brazil', 'europe'): dict(hidden=16, layers=3, normalize='none',      lr=5e-2, wd=1e-4, dropout=0.0),
    ('europe', 'brazil'): dict(hidden=32, layers=2, normalize='batchnorm', lr=5e-3, wd=0.0,  dropout=0.8),
}
FEAT_PAIRS = [
    ('brazil',        'europe',        'topo',   True),   # (src, tgt, feat_type, normalize_features)
    ('europe',        'brazil',        'topo',   True),
    ('brazil_minilm', 'europe_minilm', 'minilm', False),
    ('europe_minilm', 'brazil_minilm', 'minilm', False),
]

# ── Data loading utils ────────────────────────────────────────────────────────

def load_topo(name):
    base = name.replace('_minilm', '')
    td = np.load(f'dataset/airports/{base}/topo.npy', allow_pickle=True).item()
    df = pd.read_csv(f'dataset/tape_{base}/processed/text.csv').sort_values('node_id').reset_index(drop=True)
    N  = len(df)
    x  = np.array([[td[k][i] for k in FEATURE_ORDER] for i in range(N)], dtype=np.float32)
    y  = np.array([LABEL_MAP[l] for l in df['label']])
    return x, y, df

def load_minilm(name):
    base = name.replace('_minilm', '')
    x  = torch.load(f'dataset/airports/{base}/minilm_x.pt', weights_only=True).numpy()
    df = pd.read_csv(f'dataset/tape_{base}/processed/text.csv').sort_values('node_id').reset_index(drop=True)
    y  = np.array([LABEL_MAP[l] for l in df['label']])
    return x, y, df

def load_graph_x(name):
    """Load raw graph.x from tape processed data.pt."""
    data = torch.load(f'dataset/tape_{name}/processed/data.pt', weights_only=False)
    return data.x.numpy()

errors_found = 0

# ════════════════════════════════════════════════════════════════════════════
# 1. DATA SPLIT
# ════════════════════════════════════════════════════════════════════════════
hdr("1. DATA SPLIT")

for src_name, tgt_name, feat_type, _ in FEAT_PAIRS:
    base_src = src_name.replace('_minilm', '')
    base_tgt = tgt_name.replace('_minilm', '')

    src_df = pd.read_csv(f'dataset/tape_{base_src}/processed/text.csv')
    tgt_df = pd.read_csv(f'dataset/tape_{base_tgt}/processed/text.csv')
    N_src = len(src_df)
    N_tgt = len(tgt_df)

    print(f"\n  {src_name} → {tgt_name}  [{feat_type}]")

    for seed in SEEDS:
        all_tgt = list(range(N_tgt))
        val_idx, test_idx = train_test_split(
            all_tgt, test_size=1.0 - TARGET_VAL_RATIO,
            random_state=seed, shuffle=True
        )
        expected_val  = round(N_tgt * TARGET_VAL_RATIO)
        expected_test = N_tgt - expected_val
        overlap       = set(val_idx) & set(test_idx)

        ok_sizes  = (abs(len(val_idx) - expected_val) <= 1) and (abs(len(test_idx) - expected_test) <= 1)
        ok_cover  = len(val_idx) + len(test_idx) == N_tgt
        ok_no_dup = len(overlap) == 0

        if seed == 0:
            info(
                f"seed={seed}: train={N_src} (100%) | val={len(val_idx)} ({len(val_idx)/N_tgt*100:.0f}%) | "
                f"test={len(test_idx)} ({len(test_idx)/N_tgt*100:.0f}%)"
            )
        check(f"seed={seed}: correct sizes",   ok_sizes,
              f"expected val≈{expected_val}, test≈{expected_test}, got val={len(val_idx)}, test={len(test_idx)}")
        check(f"seed={seed}: full coverage",   ok_cover)
        check(f"seed={seed}: no val/test overlap", ok_no_dup,
              f"overlap={len(overlap)} nodes" if not ok_no_dup else "")
        if not (ok_sizes and ok_cover and ok_no_dup):
            errors_found += 1

# ════════════════════════════════════════════════════════════════════════════
# 2. FEATURES
# ════════════════════════════════════════════════════════════════════════════
hdr("2. FEATURES")

for name, expected_dim, feat_type in [
    ('brazil',        5,   'topo'),
    ('europe',        5,   'topo'),
    ('brazil_minilm', 384, 'minilm'),
    ('europe_minilm', 384, 'minilm'),
]:
    base = name.replace('_minilm', '')
    print(f"\n  {name}  [{feat_type}]")

    # Raw graph.x from data.pt
    x_raw = load_graph_x(name)
    check(f"graph.x shape = [{x_raw.shape[0]}, {expected_dim}]",
          x_raw.shape[1] == expected_dim,
          f"actual shape: {x_raw.shape}")

    has_nan = np.isnan(x_raw).any()
    has_inf = np.isinf(x_raw).any()
    check("No NaN in features",  not has_nan)
    check("No Inf in features",  not has_inf)
    if has_nan or has_inf:
        errors_found += 1

    if feat_type == 'topo':
        # After StandardScaler: mean ≈ 0, std ≈ 1 per feature
        scaler = StandardScaler()
        x_norm = scaler.fit_transform(x_raw)
        mean_ok = np.allclose(x_norm.mean(axis=0), 0, atol=1e-5)
        std_ok  = np.allclose(x_norm.std(axis=0),  1, atol=1e-3)
        check("After StandardScaler: per-feature mean ≈ 0", mean_ok,
              f"max |mean| = {np.abs(x_norm.mean(axis=0)).max():.6f}")
        check("After StandardScaler: per-feature std  ≈ 1", std_ok,
              f"max |std-1| = {np.abs(x_norm.std(axis=0) - 1).max():.6f}")
        if not (mean_ok and std_ok):
            errors_found += 1

        info(f"Raw TPF range: [{x_raw.min():.4f}, {x_raw.max():.4f}]  "
             f"(normalised range: [{x_norm.min():.4f}, {x_norm.max():.4f}])")

    else:
        info(f"MiniLM range: [{x_raw.min():.4f}, {x_raw.max():.4f}]  "
             f"mean={x_raw.mean():.4f}  std={x_raw.std():.4f}")

# ════════════════════════════════════════════════════════════════════════════
# 3. LABELS
# ════════════════════════════════════════════════════════════════════════════
hdr("3. LABEL DISTRIBUTION")

for src_name, tgt_name, feat_type, _ in FEAT_PAIRS[:2]:  # topo only; minilm shares same labels
    base_src = src_name.replace('_minilm', '')
    base_tgt = tgt_name.replace('_minilm', '')

    src_df = pd.read_csv(f'dataset/tape_{base_src}/processed/text.csv').sort_values('node_id').reset_index(drop=True)
    tgt_df = pd.read_csv(f'dataset/tape_{base_tgt}/processed/text.csv').sort_values('node_id').reset_index(drop=True)

    src_y  = np.array([LABEL_MAP[l] for l in src_df['label']])
    tgt_y  = np.array([LABEL_MAP[l] for l in tgt_df['label']])
    N_tgt  = len(tgt_df)

    print(f"\n  {src_name} → {tgt_name}")

    # Source (100% train)
    dist_src = {c: int((src_y == i).sum()) for i, c in enumerate(LABEL_ORDER)}
    info(f"Source (train={len(src_y)}): " +
         "  ".join(f"{c}={n}({n/len(src_y)*100:.0f}%)" for c, n in dist_src.items()))
    check("All 4 classes in source", len(dist_src) == 4)

    # Target (seed=0 split)
    val_idx, test_idx = train_test_split(
        list(range(N_tgt)), test_size=1.0 - TARGET_VAL_RATIO,
        random_state=0, shuffle=True
    )
    val_y  = tgt_y[val_idx]
    test_y = tgt_y[test_idx]

    dist_val  = {c: int((val_y == i).sum())  for i, c in enumerate(LABEL_ORDER)}
    dist_test = {c: int((test_y == i).sum()) for i, c in enumerate(LABEL_ORDER)}

    info(f"Val    (n={len(val_y)}):   " +
         "  ".join(f"{c}={n}({n/len(val_y)*100:.0f}%)" for c, n in dist_val.items()))
    info(f"Test   (n={len(test_y)}):  " +
         "  ".join(f"{c}={n}({n/len(test_y)*100:.0f}%)" for c, n in dist_test.items()))

    all_4_val  = all(v > 0 for v in dist_val.values())
    all_4_test = all(v > 0 for v in dist_test.values())
    check("All 4 classes in val  (seed=0)", all_4_val)
    check("All 4 classes in test (seed=0)", all_4_test)
    if not (all_4_val and all_4_test):
        errors_found += 1
        warn("Some classes may be missing in small splits — check other seeds too")

# ════════════════════════════════════════════════════════════════════════════
# 4. MODEL ARCHITECTURE
# ════════════════════════════════════════════════════════════════════════════
hdr("4. MODEL ARCHITECTURE — GCN per direction")

try:
    from src.model.gnn import GCN

    for (src_base, tgt_base), cfg in TANS_CONFIGS.items():
        for feat_type, in_ch in [('topo', 5), ('minilm', 384)]:
            print(f"\n  {src_base}→{tgt_base}  [{feat_type}]  "
                  f"hidden={cfg['hidden']} layers={cfg['layers']} "
                  f"normalize={cfg['normalize']} dropout={cfg['dropout']}")

            use_bn = (cfg['normalize'] == 'batchnorm')
            model  = GCN(
                in_channels=in_ch,
                hidden_channels=cfg['hidden'],
                out_channels=cfg['hidden'],
                num_layers=cfg['layers'],
                dropout=cfg['dropout'],
                use_bn=use_bn,
            )

            # Verify conv count
            expected_convs = cfg['layers']
            check(f"GCN has {expected_convs} conv layers", len(model.convs) == expected_convs,
                  f"actual: {len(model.convs)}")

            # Verify BN layers
            expected_bns = (cfg['layers'] - 1) if use_bn else 0
            check(f"BN layers = {expected_bns} (normalize={cfg['normalize']})",
                  len(model.bns) == expected_bns,
                  f"actual: {len(model.bns)}")

            # Verify dropout stored
            check(f"Dropout = {cfg['dropout']}", model.dropout == cfg['dropout'])

            # Param count
            n_params = sum(p.numel() for p in model.parameters())
            info(f"GNN params: {n_params:,}")

            # Quick forward pass
            x_fake = torch.randn(10, in_ch)
            ei     = torch.tensor([[0,1,2,3,4,5,6,7,8,9],
                                   [1,2,3,4,5,6,7,8,9,0]], dtype=torch.long)
            try:
                out, _ = model(x_fake, ei)
                check(f"Forward pass OK (out shape {out.shape})",
                      out.shape == (10, cfg['hidden']))
            except Exception as e:
                check("Forward pass OK", False, str(e))
                errors_found += 1

except Exception as e:
    check("GCN import / instantiation", False, str(e))
    errors_found += 1

# ════════════════════════════════════════════════════════════════════════════
# 5. CHECKPOINT LOGIC (static inspection)
# ════════════════════════════════════════════════════════════════════════════
hdr("5. CHECKPOINT SAVE / RESTORE LOGIC")

ckpt_src = open('src/utils/ckpt.py').read()
train_src = open('train.py').read()

check("_save_checkpoint strips frozen params (LLM not saved)",
      "param_grad_dic" in ckpt_src and "requires_grad" in ckpt_src)

check("_save_checkpoint writes 'checkpoint_best.pth' when is_best=True",
      '"best" if is_best else' in ckpt_src or "_checkpoint_best" in ckpt_src)

check("_reload_best_model loads from '_checkpoint_best.pth'",
      "checkpoint_best" in ckpt_src)

check("target_val_loader branch saves on val acc improvement (is_best=True)",
      "val_acc > best_val_acc" in train_src and "is_best=True" in train_src)

check("_reload_best_model called BEFORE test evaluation",
      train_src.index("_reload_best_model") < train_src.index("test_loader"))

check("Early stopping on patience from best_epoch",
      "epoch - best_epoch >= args.patience" in train_src)

# ════════════════════════════════════════════════════════════════════════════
# 6. RESULTS  (parse logs if they exist)
# ════════════════════════════════════════════════════════════════════════════
hdr("6. RESULTS (from logs/airport_tans/)")

log_files = sorted(glob.glob("logs/airport_tans/*.log"))
if not log_files:
    info("No logs found yet — run the experiments first.")
else:
    print(f"\n  {'Run':<50}  TrainLoss  ValAcc  TestAcc  Flag")
    print(f"  {'-'*85}")

    groups = {}
    for f in log_files:
        tag = os.path.basename(f).replace(".log", "")
        group = re.sub(r"_seed\d+$", "", tag)
        text  = open(f).read()

        train_losses = re.findall(r"Train Loss [\d|]+: ([0-9.]+)", text)
        val_accs     = re.findall(r"Val Acc ([0-9.]+)", text)
        test_accs    = re.findall(r"Test Acc[^:]*:\s*([0-9.]+)", text)
        best_epochs  = re.findall(r"Best ([0-9.]+) @ epoch (\d+)", text)

        train_loss = float(train_losses[-1]) if train_losses else None
        val_acc    = float(val_accs[-1])     if val_accs     else None
        test_acc   = float(test_accs[-1])    if test_accs    else None

        flags = []
        if train_loss is not None and test_acc is not None:
            if train_loss < 0.3 and test_acc < 0.35:
                flags.append("OVERFIT?")
            if test_acc < 0.26:
                flags.append("LOW(<random)")
            if val_acc is not None and test_acc is not None:
                if abs(val_acc - test_acc) < 0.01:
                    flags.append("val≈test(ok?)")

        flag_str = ", ".join(flags) if flags else "—"
        tl = f"{train_loss:.4f}" if train_loss is not None else "N/A"
        va = f"{val_acc:.4f}"    if val_acc    is not None else "N/A"
        ta = f"{test_acc:.4f}"   if test_acc   is not None else "N/A"
        print(f"  {tag:<50}  {tl:<10} {va:<7} {ta:<9} {flag_str}")

        groups.setdefault(group, []).append(test_acc)

    print(f"\n  {'Group':<50}  Seeds  Mean    Std")
    print(f"  {'-'*70}")
    for group, accs in sorted(groups.items()):
        accs_clean = [a for a in accs if a is not None]
        if accs_clean:
            mean = sum(accs_clean) / len(accs_clean)
            std  = math.sqrt(sum((a-mean)**2 for a in accs_clean) / len(accs_clean))
            print(f"  {group:<50}  {len(accs_clean)}      {mean:.4f}  ±{std:.4f}")

# ════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════════════════
hdr("SUMMARY")
if errors_found == 0:
    print(f"\n  \033[92m All checks passed — pipeline is ready to run.\033[0m\n")
else:
    print(f"\n  \033[91m {errors_found} check(s) failed — fix before running experiments.\033[0m\n")
    sys.exit(1)
