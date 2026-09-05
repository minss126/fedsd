import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from kd_necessity_diagnostic import (
    evaluate_branch_teacher_metrics,
    preserve_evaluation_rng,
)
from train import estimate_client_branch_need_gates


class ToyBYOT(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        index = x[:, 0].long()
        teacher = torch.tensor([[3.0, 0.0], [0.0, 3.0]], device=x.device)[index]
        b1 = torch.tensor([[0.0, 3.0], [3.0, 0.0]], device=x.device)[index]
        b2 = teacher.clone()
        b3 = 0.5 * teacher
        feature = x[:, :, None, None]
        return teacher, b1, b2, b3, feature, feature, feature, feature


def test_branch_metrics_distinguish_identical_and_wrong_branch():
    loader = DataLoader(
        TensorDataset(torch.tensor([[0.0], [1.0]]), torch.tensor([0, 1])),
        batch_size=2,
        shuffle=False,
    )
    model = ToyBYOT()
    metrics = evaluate_branch_teacher_metrics(
        model, loader, torch.device("cpu"), num_classes=2
    )
    assert metrics["B2"]["branch_teacher_js_normalized"] < 1e-7
    assert metrics["B2"]["need_js_x_label_advantage"] < 1e-8
    assert metrics["B1"]["branch_teacher_js_normalized"] > 0.7
    assert metrics["B1"]["teacher_label_advantage_positive"] > 0.8
    assert metrics["B1"]["need_js_x_label_advantage"] > 0.6
    assert metrics["B1"]["teacher_correct_branch_wrong_rate"] == 1.0


def test_rng_context_restores_process_and_loader_generator_states():
    generator = torch.Generator().manual_seed(17)
    loader = DataLoader(
        TensorDataset(torch.arange(8).float()[:, None], torch.arange(8)),
        batch_size=2,
        shuffle=True,
        generator=generator,
    )
    random.seed(3)
    np.random.seed(5)
    torch.manual_seed(7)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    generator_state = generator.get_state().clone()
    with preserve_evaluation_rng([loader], torch.device("cpu")):
        random.random()
        np.random.rand()
        torch.rand(1)
        list(loader)
    assert random.getstate() == python_state
    restored_numpy = np.random.get_state()
    assert restored_numpy[0] == numpy_state[0]
    assert np.array_equal(restored_numpy[1], numpy_state[1])
    assert restored_numpy[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert torch.equal(generator.get_state(), generator_state)


def test_training_need_proxy_matches_diagnostic_definitions():
    loader = DataLoader(
        TensorDataset(torch.tensor([[0.0], [1.0]]), torch.tensor([0, 1])),
        batch_size=2,
        shuffle=False,
    )
    diagnostic = evaluate_branch_teacher_metrics(
        ToyBYOT(), loader, torch.device("cpu"), num_classes=2
    )
    for proxy, metric, gain in (
        ("js", "branch_teacher_js_normalized", 1.0),
        ("advantage", "teacher_label_advantage_positive", 0.5),
        ("combined", "need_js_x_label_advantage", 1.5),
    ):
        args = type(
            "Args",
            (),
            {
                "byot_branch_need_proxy": proxy,
                "byot_branch_need_temperature": 1.0,
                "byot_branch_need_gain": gain,
                "byot_branch_need_min_gate": 0.0,
                "byot_proxy_temperature": 1.0,
            },
        )()
        gates = estimate_client_branch_need_gates(
            ToyBYOT(), loader, torch.device("cpu"), args
        )
        for index, branch in enumerate(("B1", "B2", "B3")):
            expected_raw = diagnostic[branch][metric]
            expected_gate = min(1.0, gain * expected_raw)
            assert abs(args._last_client_branch_need_stats[f"raw_{branch}"] - expected_raw) < 1e-6
            assert abs(float(gates[index]) - expected_gate) < 1e-6


def test_constant_need_gate_is_branch_and_data_independent():
    loader = DataLoader(
        TensorDataset(torch.tensor([[0.0], [1.0]]), torch.tensor([0, 1])),
        batch_size=2,
        shuffle=False,
    )
    args = type(
        "Args",
        (),
        {
            "byot_branch_need_proxy": "constant",
            "byot_branch_need_temperature": 1.0,
            "byot_branch_need_gain": 0.2,
            "byot_branch_need_min_gate": 0.0,
            "byot_proxy_temperature": 1.0,
        },
    )()
    gates = estimate_client_branch_need_gates(
        ToyBYOT(), loader, torch.device("cpu"), args
    )
    assert torch.allclose(gates, torch.full((3,), 0.2))
    assert args._last_client_branch_need_stats["sample_count"] == 0


def test_client_js_need_gate_uses_one_mean_gate_for_all_branches():
    loader = DataLoader(
        TensorDataset(torch.tensor([[0.0], [1.0]]), torch.tensor([0, 1])),
        batch_size=2,
        shuffle=False,
    )
    args = type(
        "Args",
        (),
        {
            "byot_branch_need_proxy": "js_client",
            "byot_branch_need_temperature": 1.0,
            "byot_branch_need_gain": 0.75,
            "byot_branch_need_min_gate": 0.0,
            "byot_proxy_temperature": 1.0,
        },
    )()
    gates = estimate_client_branch_need_gates(
        ToyBYOT(), loader, torch.device("cpu"), args
    )
    stats = args._last_client_branch_need_stats
    raw_mean = sum(stats[f"raw_B{index}"] for index in (1, 2, 3)) / 3.0
    expected = min(1.0, 0.75 * raw_mean)
    assert torch.allclose(gates, torch.full((3,), expected))
    assert abs(stats["raw_client_mean"] - raw_mean) < 1e-6
    assert abs(stats["gate_client"] - expected) < 1e-6
