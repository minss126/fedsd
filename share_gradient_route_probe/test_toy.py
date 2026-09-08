"""CPU smoke test for the standalone probe."""

from __future__ import annotations

import torch

from gradient_route_probe import (
    ProbeConfig,
    measure_gradient_routes,
    run_round_probe,
)


class ToyBYOT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(5, 7)
        self.layer2 = torch.nn.Linear(7, 7)
        self.layer3 = torch.nn.Linear(7, 7)
        self.layer4 = torch.nn.Linear(7, 7)
        self.fc = torch.nn.Linear(7, 3)
        self.aux1 = torch.nn.Linear(7, 3)
        self.aux2 = torch.nn.Linear(7, 3)
        self.aux3 = torch.nn.Linear(7, 3)

    def forward(self, x):
        z1 = torch.relu(self.layer1(x))
        z2 = torch.relu(self.layer2(z1))
        z3 = torch.relu(self.layer3(z2))
        z4 = torch.relu(self.layer4(z3))
        return self.fc(z4), self.aux1(z1), self.aux2(z2), self.aux3(z3)


def adapter(model, x):
    output = model(x)
    return output[0], output[1:]


def main():
    torch.manual_seed(0)
    model = ToyBYOT()
    local = {
        3: [(torch.randn(8, 5), torch.randint(0, 3, (8,)))],
        7: [(torch.randn(12, 5), torch.randint(0, 3, (12,)))],
    }
    global_reference = [
        (torch.randn(20, 5), torch.randint(0, 3, (20,)))
    ]
    report = run_round_probe(
        model_at_checkpoint=model,
        selected_client_batches=local,
        fedavg_client_weights={3: 8.0, 7: 12.0},
        global_reference_batches=global_reference,
        device="cpu",
        forward_adapter=adapter,
        shared_roots=("layer1", "layer2", "layer3", "layer4", "fc"),
        branch_prefixes={
            "B1": ("layer1",),
            "B2": ("layer1", "layer2"),
            "B3": ("layer1", "layer2", "layer3"),
            "All": ("layer1", "layer2", "layer3"),
        },
        incremental_groups={
            "Stage1": ("layer1",),
            "Stage2": ("layer2",),
            "Stage3": ("layer3",),
        },
        config=ProbeConfig(),
    )
    assert set(report["aggregate_comparisons"]) == {"B1", "B2", "B3", "All"}
    assert (
        report["aggregate_comparisons"]["B1"]
        ["local_aux_ce_to_main_ce"]["cosine"]
        is not None
    )
    assert set(report["per_client_comparisons"]) == {"3", "7"}
    assert "negative_cosine_rate" in (
        report["per_client_summary"]["B1"]["local_aux_ce_to_main_ce"]
    )
    roots = ("layer1", "layer2", "layer3", "layer4", "fc")
    routes = measure_gradient_routes(
        model=model,
        batches=local[3] + local[7],
        device="cpu",
        forward_adapter=adapter,
        shared_roots=roots,
    )
    for loss_kind in ("ce", "kd"):
        expected = {
            name: sum(
                routes[f"aux_{loss_kind}_b{branch}"][name]
                for branch in (1, 2, 3)
            )
            for name in routes[f"aux_{loss_kind}_all"]
        }
        for parameter_name, value in expected.items():
            torch.testing.assert_close(
                routes[f"aux_{loss_kind}_all"][parameter_name],
                value,
                rtol=2e-5,
                atol=2e-6,
            )
    assert all(parameter.grad is None for parameter in model.parameters())
    print("OK: standalone gradient-route probe")


if __name__ == "__main__":
    main()
