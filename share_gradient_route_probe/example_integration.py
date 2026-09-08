"""Minimal integration example.

This file is intentionally not an experiment launcher.  Call
``measure_one_round`` from the recipient's FL loop at a diagnostic round.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Mapping

import torch

from gradient_route_probe import (
    DEFAULT_RESNET_GROUPS,
    DEFAULT_RESNET_PREFIXES,
    ProbeConfig,
    byot_tuple_adapter,
    cache_loader,
    run_round_probe,
    save_report,
)


SHARED_RESNET_ROOTS = (
    "conv1",
    "bn1",
    "layer1",
    "layer2",
    "layer3",
    "layer4",
    "fc",
)


def ordered_client_batches(loader, batch_size: int = 512):
    """Use every local sample once, without shuffle or drop_last.

    The dataset's transform is left unchanged.  Use a deterministic diagnostic
    transform if exact repeatability across independently launched runs is
    required.
    """

    diagnostic_loader = torch.utils.data.DataLoader(
        loader.dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        collate_fn=getattr(loader, "collate_fn", None),
    )
    return cache_loader(diagnostic_loader)


def measure_one_round(
    global_model,
    round_start_state: Mapping[str, torch.Tensor],
    selected_client_ids,
    client_train_loaders,
    fedavg_weights,
    full_test_loader,
    completed_round: int,
    output_dir: str | Path,
    device: str = "cuda:0",
):
    """Example callback executed for a small set of diagnostic rounds.

    Important: ``round_start_state`` must be saved before local optimization.
    The locally trained client models are not used for these six gradients.
    """

    probe_model = copy.deepcopy(global_model).to(device)
    probe_model.load_state_dict(round_start_state, strict=True)

    selected_batches = {
        client_id: ordered_client_batches(client_train_loaders[client_id])
        for client_id in selected_client_ids
    }
    selected_weights = {
        client_id: float(fedavg_weights[client_id])
        for client_id in selected_client_ids
    }
    global_batches = cache_loader(full_test_loader)  # full reference set

    report = run_round_probe(
        model_at_checkpoint=probe_model,
        selected_client_batches=selected_batches,
        fedavg_client_weights=selected_weights,
        global_reference_batches=global_batches,
        device=device,
        forward_adapter=byot_tuple_adapter,
        shared_roots=SHARED_RESNET_ROOTS,
        branch_prefixes=DEFAULT_RESNET_PREFIXES,
        incremental_groups=DEFAULT_RESNET_GROUPS,
        config=ProbeConfig(
            branch_indices=(0, 1, 2),
            branch_reduction="sum",
            branch_ce_label_smoothing=0.0,
            teacher_temperature=1.0,
            student_temperature=1.0,
            kd_scale_mode="student_t_squared",
        ),
    )
    save_report(report, Path(output_dir) / f"gradient_routes_r{completed_round}.json")
    return report


# If the model returns a dict, replace byot_tuple_adapter with something like:
#
# def my_adapter(model, x):
#     output = model(x)
#     return output["final"], (output["b1"], output["b2"], output["b3"])
