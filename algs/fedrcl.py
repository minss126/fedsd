import torch
import torch.nn as nn
import torch.nn.functional as F

class RCLloss(nn.Module):

    def __init__(self):
        super(RCLloss, self).__init__()

    def forward(self, cosine_sim_matrix, x, target, args):
        target_expand = target.unsqueeze(1)
        mask_same = (target_expand == target_expand.t())
        valid_mask = ~torch.eye(x.size(0), dtype=torch.bool, device=args.device)
        mask_same = mask_same & valid_mask
        mask_diff = (target_expand != target_expand.t())

        pos_sim = torch.where(mask_same, cosine_sim_matrix, torch.tensor(2.0, device=args.device))
        hard_positive, _ = pos_sim.min(dim=1)
        
        pos_sim_max = torch.where(mask_same, cosine_sim_matrix, torch.tensor(-2.0, device=args.device))
        easy_positive, _ = pos_sim_max.max(dim=1)
        
        neg_sim = torch.where(mask_diff, cosine_sim_matrix, torch.tensor(2.0, device=args.device))
        hard_negative, _ = neg_sim.min(dim=1)
        
        loss_pull = F.binary_cross_entropy_with_logits(hard_positive, torch.ones_like(hard_positive))
        loss_push_pos = F.binary_cross_entropy_with_logits(easy_positive, torch.zeros_like(easy_positive))
        loss_push_neg = F.binary_cross_entropy_with_logits(hard_negative, torch.zeros_like(hard_negative))
        loss_push = loss_push_pos + loss_push_neg
        
        tau = args.tau_rcl
        beta = args.beta_rcl
        threshold = args.threshold_rcl

        N = target.size(0)

        t = target.unsqueeze(1)                                     
        mask_same = (t == t.t()).to(torch.bool)                     
        eye = torch.eye(N, dtype=torch.bool, device=args.device)
        base_pos_mask = mask_same & ~eye                             

        sim_threshold_mask = (cosine_sim_matrix >= threshold)       

        pos_mask = base_pos_mask & sim_threshold_mask              

        exp_sim = torch.exp(cosine_sim_matrix / tau)           

        sum_pos = (exp_sim * pos_mask.float()).sum(dim=1)           

        self_term = torch.exp(torch.tensor(1.0, device=args.device) / tau)
        sum_all = sum_pos + self_term                               

        loss_penalty = beta * torch.log(sum_all)                       
        loss_penalty = loss_penalty.mean()                                    


        rcl_loss = loss_pull + loss_push + loss_penalty

        return rcl_loss
