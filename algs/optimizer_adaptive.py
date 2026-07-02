import torch
import torch.nn.functional as F

__all__ = ["ExpSAM"]

class ExpSAM(torch.optim.Optimizer):
    def __init__(
        self, params, ref_params, base_optimizer, rho=0.05, adaptive=False, **kwargs
    ):
        # assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(ExpSAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

        self.ref_param_groups = []
        ref_param_groups = list(ref_params)

        if not isinstance(ref_param_groups[0], dict):
            ref_param_groups = [{"params": ref_param_groups}]

        for ref_param_group in ref_param_groups:
            self.add_ref_param_group(ref_param_group)

        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group, ref_group in zip(self.param_groups, self.ref_param_groups):
            scale = group["rho"] / (grad_norm + 1e-12)

            for p, ref_p in zip(group["params"], ref_group["params"]):
                if p.grad is None:
                    try:
                        self.state[p]["old_p"] = p.data.clone()
                    except:
                        pass

                    continue

                # avg_mag = torch.abs(p - ref_p).mean()

                self.state[p]["old_p"] = p.data.clone()
                e_w = F.normalize((p - ref_p).abs(), 2, dim=0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert (
            closure is not None
        ), "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(
            closure
        )  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][
            0
        ].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack(
                [
                    (1.0 * p.grad).norm(p=2).to(shared_device)
                    for group, ref_group in zip(
                        self.param_groups, self.ref_param_groups
                    )
                    for p, ref_p in zip(group["params"], ref_group["params"])
                    if p.grad is not None
                ]
            ),
            p=2,
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

    def add_ref_param_group(self, param_group):
        params = param_group["params"]

        if isinstance(params, torch.Tensor):
            param_group["params"] = [params]
        else:
            param_group["params"] = list(params)

        for name, default in self.defaults.items():
            param_group.setdefault(name, default)

        params = param_group["params"]

        param_set = set()
        for group in self.ref_param_groups:
            param_set.update(set(group["params"]))

        if not param_set.isdisjoint(set(param_group["params"])):
            raise ValueError("some parameters appear in more than one parameter group")

        self.ref_param_groups.append(param_group)