import sys
import torch
import wandb
import copy
import gc
import os
import json
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split as _sklearn_split
from sklearn.preprocessing import StandardScaler


from src.utils.seed import seed_everything
from src.utils.lr_schedule import adjust_learning_rate
from torch.nn.utils import clip_grad_norm_
from src.config import parse_args_llama
from src.utils.ckpt import _save_checkpoint, _reload_best_model
from src.model import load_model, llama_model_path
from src.dataset import load_dataset
from src.utils.evaluate import eval_funcs
from src.utils.collate import collate_funcs


def save_run_report(args, run_name, test_dataset_name, seed,
                    best_epoch, best_val_acc, best_val_loss,
                    test_acc, train_size, val_size, target_val_size, test_size,
                    trainable_params, all_param):
    """Save a detailed run report to a txt file after each training."""

    report_dir = os.path.join(args.output_dir, "reports")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, f"{run_name}.txt")

    ckpt_path = os.path.join(args.output_dir, f"best_{run_name}.pth")

    lines = [
        "=" * 60,
        f"RUN REPORT — {run_name}",
        f"Timestamp     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,

        "\n--- Transfer Setting ---",
        f"Source dataset: {args.dataset}",
        f"Target dataset: {test_dataset_name}",
        f"Cross-dataset : {test_dataset_name != args.dataset}",
        f"Seed          : {seed}",

        "\n--- Data Split ---",
        f"Train nodes   : {train_size}  (source graph, 100%)",
        f"Val nodes     : {val_size}    (source val, same-dataset mode)",
        f"Target val    : {target_val_size}  (target graph, 20%)",
        f"Test nodes    : {test_size}   (target graph, 80%)",

        "\n--- Model Info ---",
        f"Model name    : {args.model_name}",
        f"LLM           : {args.llm_model_name}",
        f"GNN           : {args.gnn_model_name}",
        f"GNN hidden dim: {args.gnn_hidden_dim}",
        f"GNN out dim   : {args.gnn_out_dim}",
        f"GNN layers    : {args.gnn_num_layers}",
        f"Trainable params: {trainable_params} / {all_param} "
        f"({100 * trainable_params / all_param:.2f}%)",

        "\n--- Training Hyperparameters ---",
        f"Epochs        : {args.num_epochs}",
        f"Batch size    : {args.batch_size}",
        f"Learning rate : {args.lr}",
        f"Weight decay  : {args.wd}",
        f"Dropout       : {getattr(args, 'dropout', 'N/A')}",
        f"Patience      : {args.patience}",
        f"Grad steps    : {args.grad_steps}",
        f"Normalize feat: {getattr(args, 'normalize_features', False)}",
        f"Target val ratio: {getattr(args, 'target_val_ratio', 0.2)}",

        "\n--- Results ---",
        f"Best epoch    : {best_epoch}",
        f"Best val acc  : {best_val_acc:.4f}" if best_val_acc >= 0 else "Best val acc  : N/A (no val set)",
        f"Best val loss : {best_val_loss:.4f}" if best_val_loss < float('inf') else "Best val loss : N/A",
        f"Test accuracy : {test_acc:.4f}  ({test_acc*100:.2f}%)",

        "\n--- Checkpoint ---",
        f"Weights saved : {ckpt_path}",

        "=" * 60,
    ]

    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    # Also append to a global summary file
    summary_path = os.path.join(report_dir, "all_runs_summary.txt")
    with open(summary_path, "a") as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
                f"{run_name} | "
                f"best_epoch={best_epoch} | "
                f"val_acc={best_val_acc:.4f} | "
                f"test_acc={test_acc:.4f} ({test_acc*100:.2f}%)\n")

    print(f"\n[Report saved] {report_path}")
    print(f"[Summary updated] {summary_path}")
    return report_path


def main(args):
    seed = args.seed

    test_dataset_name = args.test_dataset if args.test_dataset else args.dataset
    run_name = f"{args.dataset}→{test_dataset_name}_{args.model_name}_seed{seed}"

    wandb.init(project=f"{args.project}", name=run_name, config=args)
    seed_everything(seed=args.seed)
    print(args)

    # ── Train / Val dataset ──────────────────────────────────────────────────
    dataset   = load_dataset[args.dataset]()
    idx_split = dataset.get_idx_split()

    _cli_flags = {a.lstrip('-').replace('-', '_') for a in sys.argv if a.startswith('--')}
    for _attr in ('gnn_hidden_dim', 'gnn_out_dim', 'gnn_num_layers'):
        if hasattr(dataset, _attr) and _attr not in _cli_flags:
            setattr(args, _attr, getattr(dataset, _attr))

    cross_dataset = test_dataset_name != args.dataset

    if cross_dataset:
        test_src        = load_dataset[test_dataset_name]()
        test_collate_fn = collate_funcs[test_dataset_name](test_src.graph)
    else:
        test_src        = dataset
        test_collate_fn = collate_funcs[args.dataset](dataset.graph)

    if args.normalize_features:
        scaler = StandardScaler()
        dataset.graph.x = torch.tensor(
            scaler.fit_transform(dataset.graph.x.cpu().numpy()), dtype=torch.float32
        )
        if cross_dataset:
            test_src.graph.x = torch.tensor(
                scaler.transform(test_src.graph.x.cpu().numpy()), dtype=torch.float32
            )

    collate_fn = collate_funcs[args.dataset](dataset.graph)
    if cross_dataset:
        test_collate_fn = collate_funcs[test_dataset_name](test_src.graph)
    else:
        test_collate_fn = collate_fn

    if cross_dataset:
        train_indices = list(range(len(dataset)))
        val_indices   = []
        all_target    = list(range(len(test_src)))
        if args.target_val_ratio > 0:
            target_val_indices, test_indices = _sklearn_split(
                all_target,
                test_size=1.0 - args.target_val_ratio,
                random_state=seed,
                shuffle=True,
            )
        else:
            target_val_indices = []
            test_indices       = all_target
    else:
        train_indices      = idx_split['train']
        val_indices        = idx_split['val']
        target_val_indices = []
        test_indices       = idx_split['test']

    train_dataset      = [dataset[i] for i in train_indices]
    val_dataset        = [dataset[i] for i in val_indices]
    target_val_dataset = [test_src[i] for i in target_val_indices]
    test_dataset       = [test_src[i] for i in test_indices]

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              drop_last=True, pin_memory=True, shuffle=True,
                              collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset, batch_size=args.eval_batch_size,
                              drop_last=False, pin_memory=True, shuffle=False,
                              collate_fn=collate_fn) if val_dataset else None
    target_val_loader = DataLoader(target_val_dataset, batch_size=args.eval_batch_size,
                                   drop_last=False, pin_memory=True, shuffle=False,
                                   collate_fn=test_collate_fn) if target_val_dataset else None
    test_loader  = DataLoader(test_dataset, batch_size=args.eval_batch_size,
                              drop_last=False, pin_memory=True, shuffle=False,
                              collate_fn=test_collate_fn)

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | "
          f"Target Val: {len(target_val_dataset)} | "
          f"Test ({test_dataset_name}): {len(test_dataset)}")

    # ── Model ────────────────────────────────────────────────────────────────
    args.llm_model_path = llama_model_path[args.llm_model_name]
    model = load_model[args.model_name](
        graph=dataset.graph, graph_type=dataset.graph_type,
        prompt=dataset.prompt, args=args,
    )

    # ── Optimizer ────────────────────────────────────────────────────────────
    params = [p for _, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{'params': params, 'lr': args.lr, 'weight_decay': args.wd}],
        betas=(0.9, 0.95),
    )
    trainable_params, all_param = model.print_trainable_params()
    print(f"trainable: {trainable_params} / {all_param} "
          f"({100 * trainable_params / all_param:.2f}%)")

    # ── Training loop ────────────────────────────────────────────────────────
    num_training_steps = args.num_epochs * len(train_loader)
    progress_bar = tqdm(range(num_training_steps))
    best_val_loss = float('inf')
    best_val_acc  = -1.0
    best_epoch    = 0

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss, accum_loss = 0., 0.

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()

            clip_grad_norm_(optimizer.param_groups[0]['params'], 0.1)

            if (step + 1) % args.grad_steps == 0:
                adjust_learning_rate(optimizer.param_groups[0],
                                     step / len(train_loader) + epoch, args)

            optimizer.step()
            epoch_loss += loss.item()
            accum_loss += loss.item()

            if (step + 1) % args.grad_steps == 0:
                lr = optimizer.param_groups[0]['lr']
                wandb.log({'Lr': lr})
                wandb.log({'Accum Loss': accum_loss / args.grad_steps})
                accum_loss = 0.

            progress_bar.update(1)

        print(f"Epoch {epoch}|{args.num_epochs}: Train Loss {epoch_loss / len(train_loader):.4f}")
        wandb.log({'Train Loss (Epoch Mean)': epoch_loss / len(train_loader)})

        # ── Validation ───────────────────────────────────────────────────────
        if val_loader is not None:
            val_loss = 0.
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    val_loss += model(batch).item()
            val_loss /= len(val_loader)
            print(f"Epoch {epoch}|{args.num_epochs}: Val Loss {val_loss:.4f}  "
                  f"Best {best_val_loss:.4f} @ epoch {best_epoch}")
            wandb.log({'Val Loss': val_loss})

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch    = epoch
                _save_checkpoint(model, optimizer, epoch, args, is_best=True)

            if epoch - best_epoch >= args.patience:
                print(f'Early stop at epoch {epoch}')
                break

        elif target_val_loader is not None:
            model.eval()
            val_output = []
            with torch.no_grad():
                for batch in target_val_loader:
                    val_output.append(model.inference(batch))
            val_acc = eval_funcs[test_dataset_name](
                val_output,
                f'{args.output_dir}/_val_{run_name}_epoch{epoch}.csv',
            )
            print(f"Epoch {epoch}|{args.num_epochs}: Val Acc {val_acc:.4f}  "
                  f"Best {best_val_acc:.4f} @ epoch {best_epoch}")
            wandb.log({'Val Acc': val_acc})

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch   = epoch
                _save_checkpoint(model, optimizer, epoch, args, is_best=True)

            if epoch - best_epoch >= args.patience:
                print(f'Early stop at epoch {epoch}')
                break

        else:
            best_epoch = epoch
            _save_checkpoint(model, optimizer, epoch, args, is_best=True)

    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()

    # ── Test ─────────────────────────────────────────────────────────────────
    model = _reload_best_model(model, args)
    model.eval()
    eval_output = []
    for batch in tqdm(test_loader, desc='Test inference'):
        with torch.no_grad():
            eval_output.append(model.inference(batch))

    path = (f'{args.output_dir}/{args.dataset}→{test_dataset_name}_'
            f'{args.model_name}_{args.llm_model_name}_{args.gnn_model_name}_seed{seed}.csv')
    acc = eval_funcs[test_dataset_name](eval_output, path)
    print(f'Test Acc ({test_dataset_name}): {acc:.4f}')
    wandb.log({'Test Acc': acc})

    # ── Save run report ──────────────────────────────────────────────────────
    save_run_report(
        args=args,
        run_name=run_name,
        test_dataset_name=test_dataset_name,
        seed=seed,
        best_epoch=best_epoch,
        best_val_acc=best_val_acc,
        best_val_loss=best_val_loss,
        test_acc=acc,
        train_size=len(train_dataset),
        val_size=len(val_dataset),
        target_val_size=len(target_val_dataset),
        test_size=len(test_dataset),
        trainable_params=trainable_params,
        all_param=all_param,
    )


if __name__ == "__main__":
    args = parse_args_llama()
    main(args)
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    gc.collect()