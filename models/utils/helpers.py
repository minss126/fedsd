"""Checkpoint-shape helpers used by the upstream CCT factories."""

import logging
import math

import torch
import torch.nn.functional as F


_LOGGER = logging.getLogger("train")


def resize_pos_embed(posemb, posemb_new, num_tokens=1):
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    grid_size_old = int(math.sqrt(len(posemb_grid)))
    grid_size_new = int(math.sqrt(ntok_new))
    posemb_grid = posemb_grid.reshape(
        1, grid_size_old, grid_size_old, -1
    ).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(
        posemb_grid,
        size=(grid_size_new, grid_size_new),
        mode="bilinear",
        align_corners=False,
    )
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(
        1, grid_size_new * grid_size_new, -1
    )
    return torch.cat([posemb_tok, posemb_grid], dim=1)


def pe_check(model, state_dict, pe_key="classifier.positional_emb"):
    if (
        pe_key is not None
        and pe_key in state_dict
        and pe_key in model.state_dict()
        and model.state_dict()[pe_key].shape != state_dict[pe_key].shape
    ):
        state_dict[pe_key] = resize_pos_embed(
            state_dict[pe_key],
            model.state_dict()[pe_key],
            num_tokens=model.classifier.num_tokens,
        )
    return state_dict


def fc_check(model, state_dict, fc_key="classifier.fc"):
    for key in (f"{fc_key}.weight", f"{fc_key}.bias"):
        if (
            key in state_dict
            and key in model.state_dict()
            and model.state_dict()[key].shape != state_dict[key].shape
        ):
            _LOGGER.warning("Replacing %s because the class count changed.", key)
            state_dict[key] = model.state_dict()[key]
    return state_dict

