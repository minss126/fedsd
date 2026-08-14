#!/usr/bin/env python3
"""Train a branch-inactive centralized checkpoint for local-n diagnostics."""

import argparse
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_utils import get_global_dataset
from models.resnet_byot import multi_resnet18_kd


NUM_CLASSES = {"cifar10": 10, "cifar100": 100}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(NUM_CLASSES), required=True)
    parser.add_argument("--datadir", default="./data")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--test_batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--eta_min", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--eval_interval", type=int, default=50,
        help="Evaluate every N epochs for progress logging; 0 evaluates only at the end.",
    )
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_torch_save(payload, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output.parent, delete=False, prefix=f".{output.name}.", suffix=".tmp"
    ) as handle:
        temporary = handle.name
    try:
        torch.save(payload, temporary)
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    loss_sum, correct, examples = 0.0, 0, 0
    for x, target in dataloader:
        x = x.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True).long()
        _, logits = model.forward_teacher(x)
        loss_sum += float(F.cross_entropy(logits, target, reduction="sum").item())
        correct += int(logits.argmax(dim=1).eq(target).sum().item())
        examples += int(target.numel())
    return {
        "loss": loss_sum / max(examples, 1),
        "accuracy_pct": 100.0 * correct / max(examples, 1),
        "samples": examples,
    }


def main():
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable.")
    seed_everything(args.seed)
    device = torch.device(args.device)

    # Attributes consumed by get_global_dataset.
    args.in_channels = 3
    args.num_classes = NUM_CLASSES[args.dataset]
    args.cifar100_class_count = 0
    args.cifar100_subset_seed = 0
    train_dataset, _, test_dataset = get_global_dataset(args)

    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed + 1)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        generator=train_generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = multi_resnet18_kd(
        num_classes=args.num_classes, in_channels=args.in_channels
    ).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.eta_min
    )

    started = time.time()
    optimizer_steps = 0
    examples_processed = 0
    history = []
    for epoch in range(args.epochs):
        model.train()
        loss_sum, examples = 0.0, 0
        for x, target in train_loader:
            x = x.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True).long()
            optimizer.zero_grad(set_to_none=True)
            _, logits = model.forward_teacher(x)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * int(target.numel())
            examples += int(target.numel())
            optimizer_steps += 1
            examples_processed += int(target.numel())
        scheduler.step()

        completed_epochs = epoch + 1
        should_eval = (
            completed_epochs == args.epochs
            or (args.eval_interval > 0 and completed_epochs % args.eval_interval == 0)
        )
        record = {
            "epoch": completed_epochs,
            "train_loss": loss_sum / max(examples, 1),
            "lr_after_epoch": float(scheduler.get_last_lr()[0]),
        }
        if should_eval:
            record["test"] = evaluate(model, test_loader, device)
        history.append(record)
        if completed_epochs == 1 or should_eval:
            suffix = ""
            if "test" in record:
                suffix = f" test_acc={record['test']['accuracy_pct']:.2f}%"
            print(
                f"epoch={completed_epochs:04d}/{args.epochs:04d} "
                f"train_loss={record['train_loss']:.6f} "
                f"lr={record['lr_after_epoch']:.6g}{suffix}",
                flush=True,
            )

    final_test = history[-1].get("test") or evaluate(model, test_loader, device)
    checkpoint_args = vars(args).copy()
    checkpoint_args.update({
        "model": "resnet18_byot",
        "checkpoint_source": "centralized_teacher_only",
        "branch_objectives_active": False,
        "train_samples": int(len(train_dataset)),
        "official_test_samples": int(len(test_dataset)),
    })
    payload = {
        "format_version": 1,
        "checkpoint_role": "centralized_teacher_only_final",
        "round": -1,
        "completed_rounds": 0,
        "completed_epochs": int(args.epochs),
        "args": checkpoint_args,
        "global_model": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "optimizer_steps": int(optimizer_steps),
        "examples_processed": int(examples_processed),
        "final_test": final_test,
        "history": history,
        "wall_time_seconds": float(time.time() - started),
    }
    atomic_torch_save(payload, args.output)
    print(
        json.dumps({
            "saved": os.path.abspath(args.output),
            "epochs": args.epochs,
            "optimizer_steps": optimizer_steps,
            "examples_processed": examples_processed,
            "final_test": final_test,
        }, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
