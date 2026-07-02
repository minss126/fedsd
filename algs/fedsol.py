import torch
from algs.optimizer_fixed import SAM
from algs.optimizer_adaptive import ExpSAM
from torch.nn.modules.batchnorm import _BatchNorm

def freeze_head(model):
    classifier_name = list(model.named_children())[-1][0]

    for name, params in model.named_parameters():
        if classifier_name in name:
            params.requires_grad = False
        else:
            params.requires_grad = True

def freeze_body(model):
    classifier_name = list(model.named_children())[-1][0]

    for name, params in model.named_parameters():
        if classifier_name in name:
            params.requires_grad = True
        else:
            params.requires_grad = False

def zerograd_head(model):
    classifier_name = list(model.named_children())[-1][0]

    # Set gradients of head to zero
    for name, params in model.named_parameters():
        if classifier_name in name:
            params.grad = torch.zeros_like(params)
        else:
            pass

def zerograd_body(model):
    classifier_name = list(model.named_children())[-1][0]

    # Set gradients of body to zero
    for name, params in model.named_parameters():
        if classifier_name in name:
            pass
        else:
            params.grad = torch.zeros_like(params)

def get_sam_optimizer_fixed(net, base_optimizer, args):
    optim_params = base_optimizer.state_dict()
    lr = optim_params["param_groups"][0]["lr"]
    momentum = optim_params["param_groups"][0]["momentum"]
    weight_decay = optim_params["param_groups"][0]["weight_decay"]
    sam_optimizer = SAM(
        net.parameters(),
        base_optimizer=torch.optim.SGD,
        rho=args.rho,
        adaptive=False,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    return sam_optimizer

def get_sam_optimizer_adaptive(net, dg_model, base_optimizer, args):
    optim_params = base_optimizer.state_dict()
    lr = optim_params["param_groups"][0]["lr"]
    momentum = optim_params["param_groups"][0]["momentum"]
    weight_decay = optim_params["param_groups"][0]["weight_decay"]
    sam_optimizer = ExpSAM(
        net.parameters(),
        dg_model.parameters(),
        base_optimizer=torch.optim.SGD,
        rho=args.rho,
        adaptive=False,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    return sam_optimizer

def unfreeze(model):
    for name, params in model.named_parameters():
        params.requires_grad = True

def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)


def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)