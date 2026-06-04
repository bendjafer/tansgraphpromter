import sys
import torch
import wandb
import copy
import gc
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.utils.seed import seed_everything
from src.utils.lr_schedule import adjust_learning_rate
from torch.nn.utils import clip_grad_norm_
from src.config import parse_args_llama
from src.utils.ckpt import _save_checkpoint, _reload_best_model
from src.model import load_model, llama_model_path
from src.dataset import load_dataset
from src.utils.evaluate import eval_funcs
from src.utils.collate import collate_funcs


def main(args):
    seed = args.seed

    # Resolve test dataset: defaults to the training dataset when not specified.
    test_dataset_name = args.test_dataset if args.test_dataset else args.dataset
    run_name = f"{args.dataset}→{test_dataset_name}_{args.model_name}_seed{seed}"

    wandb.init(project=f"{args.project}", name=run_name, config=args)
    seed_everything(seed=args.seed)
    print(args)

    # ── Train / Val dataset ──────────────────────────────────────────────────
    dataset   = load_dataset[args.dataset]()
    idx_split = dataset.get_idx_split()

    # If the dataset recommends GNN dims for its feature size, apply them —
    # but only when the user has NOT explicitly passed those flags on the CLI.
    _cli_flags = {a.lstrip('-').replace('-', '_') for a in sys.argv if a.startswith('--')}
    for _attr in ('gnn_hidden_dim', 'gnn_out_dim', 'gnn_num_layers'):
        if hasattr(dataset, _attr) and _attr not in _cli_flags:
            setattr(args, _attr, getattr(dataset, _attr))

    train_dataset = [dataset[i] for i in idx_split['train']]
    val_dataset   = [dataset[i] for i in idx_split['val']]

    collate_fn = collate_funcs[args.dataset](dataset.graph)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              drop_last=True, pin_memory=True, shuffle=True,
                              collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size,
                              drop_last=False, pin_memory=True, shuffle=False,
                              collate_fn=collate_fn)

    # ── Test dataset (may differ from training dataset) ──────────────────────
    if test_dataset_name == args.dataset:
        test_src        = dataset
        test_collate_fn = collate_fn
    else:
        test_src        = load_dataset[test_dataset_name]()
        test_collate_fn = collate_funcs[test_dataset_name](test_src.graph)

    # Cross-dataset: evaluate on ALL nodes of the target dataset (none were seen
    # during training, so there is no data-leakage reason to restrict to a split).
    # Same-dataset: use only the held-out test split as usual.
    if test_dataset_name != args.dataset:
        test_indices = list(range(len(test_src)))
    else:
        test_indices = test_src.get_idx_split()['test']

    test_dataset = [test_src[i] for i in test_indices]
    test_loader  = DataLoader(test_dataset, batch_size=args.eval_batch_size,
                              drop_last=False, pin_memory=True, shuffle=False,
                              collate_fn=test_collate_fn)

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | "
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


if __name__ == "__main__":
    args = parse_args_llama()
    main(args)
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    gc.collect()
