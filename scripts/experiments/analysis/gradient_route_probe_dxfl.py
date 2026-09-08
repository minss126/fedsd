"""DXFL integration for the standalone CE/KD gradient-route probe."""

from __future__ import annotations

import itertools
import os
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from share_gradient_route_probe.gradient_route_probe import (
    DEFAULT_RESNET_GROUPS,
    DEFAULT_RESNET_PREFIXES,
    ProbeConfig,
    byot_tuple_adapter,
    preserve_rng,
    run_round_probe,
    save_report,
)


SHARED_BRANCH_REACHABLE_ROOTS = (
    "conv1",
    "bn1",
    "layer1",
    "layer2",
    "layer3",
)


def should_run_gradient_route_probe(args, completed_round: int) -> bool:
    if not getattr(args, "log_gradient_routes", False):
        return False
    raw_rounds = str(getattr(args, "gradient_route_probe_rounds", "") or "").strip()
    if raw_rounds:
        requested = {int(token.strip()) for token in raw_rounds.split(",") if token.strip()}
        return int(completed_round) in requested
    interval = int(getattr(args, "gradient_route_probe_interval", 0))
    return interval > 0 and completed_round > 0 and completed_round % interval == 0


def _diagnostic_loader(source_loader, batch_size: int):
    """Use every client sample once while preserving its configured transform."""

    return DataLoader(
        source_loader.dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=getattr(source_loader, "collate_fn", None),
    )


def _limit_batches(loader, max_batches: int):
    if max_batches <= 0:
        return loader
    return itertools.islice(loader, max_batches)


@contextmanager
def _temporarily_enable_shared_gradients(model):
    states = {parameter: parameter.requires_grad for parameter in model.parameters()}
    try:
        for name, parameter in model.named_parameters():
            if any(
                name == root or name.startswith(root + ".")
                for root in SHARED_BRANCH_REACHABLE_ROOTS
            ):
                parameter.requires_grad_(True)
        yield
    finally:
        for parameter, required_grad in states.items():
            parameter.requires_grad_(required_grad)


def run_dxfl_gradient_route_probe(
    *,
    args,
    logger,
    log_file_name: str,
    completed_round: int,
    checkpoint_model,
    selected_client_loaders,
    fedavg_weights,
    global_test_loader,
    device,
):
    """Measure analyses 1--3 and write one JSON file for this checkpoint."""

    if getattr(args, "model", "") != "resnet18_byot":
        raise ValueError("--log_gradient_routes currently requires --model resnet18_byot")

    client_ids = list(selected_client_loaders)
    requested_clients = int(getattr(args, "gradient_route_probe_client_count", 0))
    if requested_clients > 0:
        client_ids = client_ids[:requested_clients]
    if not client_ids:
        raise ValueError("Gradient-route probe received no clients")

    weights_by_client = {
        client_id: float(fedavg_weights[index])
        for index, client_id in enumerate(selected_client_loaders)
        if client_id in client_ids
    }
    batch_size = max(1, int(getattr(args, "gradient_route_probe_batch_size", 64)))
    local_max_batches = max(0, int(getattr(args, "gradient_route_local_max_batches", 0)))
    global_max_batches = max(0, int(getattr(args, "gradient_route_global_max_batches", 0)))
    client_batches = {
        client_id: _limit_batches(
            _diagnostic_loader(selected_client_loaders[client_id], batch_size),
            local_max_batches,
        )
        for client_id in client_ids
    }
    global_batches = _limit_batches(
        _diagnostic_loader(global_test_loader, batch_size), global_max_batches
    )

    temperature = float(getattr(args, "gradient_route_temperature", 0.0))
    if temperature <= 0.0:
        temperature = float(getattr(args, "temperature", 1.0))
    config = ProbeConfig(
        branch_indices=(0, 1, 2),
        branch_reduction=str(getattr(args, "gradient_route_branch_reduction", "sum")),
        branch_ce_label_smoothing=float(
            getattr(args, "byot_branch_ce_label_smoothing", 0.0)
        ),
        teacher_temperature=temperature,
        student_temperature=temperature,
        kd_scale_mode="student_t_squared",
    )

    output_root = str(getattr(args, "gradient_route_output_dir", "") or "").strip()
    if not output_root:
        output_root = os.path.join(args.logdir, f"{log_file_name}_gradient_routes")
    output_path = Path(output_root) / f"round_{completed_round:04d}.json"
    if output_path.exists() and not getattr(args, "gradient_route_overwrite", False):
        logger.info(f"Gradient-route probe already exists, keeping: {output_path}")
        return str(output_path)

    logger.info(
        "Gradient-route probe start: "
        f"checkpoint_round={completed_round}, clients={len(client_ids)}, "
        f"temperature={temperature}, batch_size={batch_size}, "
        f"local_max_batches={local_max_batches or 'all'}, "
        f"global_max_batches={global_max_batches or 'all'}"
    )
    diagnostic_seed = (
        int(args.seed) * 1_000_003 + int(completed_round) * 97_409 + 53
    ) % (2**31 - 1)
    # A round/seed-specific diagnostic RNG makes stochastic train transforms
    # identical across paired CE/KD variants and restores the training stream.
    with preserve_rng():
        random.seed(diagnostic_seed)
        np.random.seed(diagnostic_seed)
        torch.manual_seed(diagnostic_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(diagnostic_seed)
        with _temporarily_enable_shared_gradients(checkpoint_model):
            report = run_round_probe(
                model_at_checkpoint=checkpoint_model,
                selected_client_batches=client_batches,
                fedavg_client_weights=weights_by_client,
                global_reference_batches=global_batches,
                device=device,
                forward_adapter=byot_tuple_adapter,
                shared_roots=SHARED_BRANCH_REACHABLE_ROOTS,
                branch_prefixes=DEFAULT_RESNET_PREFIXES,
                incremental_groups=DEFAULT_RESNET_GROUPS,
                config=config,
            )
    report["experiment"] = {
        "dataset": str(args.dataset),
        "partition": str(args.partition),
        "beta": float(args.beta),
        "seed": int(args.seed),
        "algorithm": str(args.alg),
        "model": str(args.model),
        "completed_aggregation_round": int(completed_round),
        "checkpoint_stage": "post_aggregation",
        "local_reference": (
            "full selected-client train subsets with train transform"
            if local_max_batches == 0
            else f"first {local_max_batches} ordered batches per selected client"
        ),
        "global_reference": (
            "full official test set"
            if global_max_batches == 0
            else f"first {global_max_batches} ordered official-test batches"
        ),
        "local_dataset_sizes": {
            str(client_id): int(len(selected_client_loaders[client_id].dataset))
            for client_id in client_ids
        },
        "local_max_batches": int(local_max_batches),
        "global_reference_samples": int(len(global_test_loader.dataset)),
        "global_max_batches": int(global_max_batches),
        "diagnostic_batch_size": int(batch_size),
        "diagnostic_seed": int(diagnostic_seed),
        "training_branch_objective": str(
            getattr(args, "byot_branch_objective", "")
        ),
        "training_byot_alpha": float(getattr(args, "byot_alpha", 0.0)),
        "training_byot_beta": float(getattr(args, "byot_beta", 0.0)),
        "note": (
            "Official-test labels are used only for post-hoc gradient analysis, "
            "never for optimization or adaptive gating. Local/global comparisons "
            "include the operational train-transform versus test-transform difference."
        ),
    }
    save_report(report, output_path)
    logger.info(f"Gradient-route probe complete: {output_path}")
    del report
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
    return str(output_path)
