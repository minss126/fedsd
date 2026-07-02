import datetime
import os
import random
import numpy as np
import torch
import torch.nn as nn
import logging
import json
import copy
import time
import torch.nn.functional as F  # Softmax 사용을 위해 추가

# 모델 임포트
from models import resnet_cifar
from models import mobilenet_v2
from models import simplecnn

try:
    from resnet_byot import multi_resnet18_kd, multi_resnet50_kd
except ImportError:
    try:
        from models.resnet_byot import multi_resnet18_kd, multi_resnet50_kd
    except ImportError:
        print("[Warning] resnet_byot not found.")

def convert_bn_to_gn(module, num_groups=8):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
            setattr(module, name, gn)
        else:
            convert_bn_to_gn(child, num_groups)
    return module

def init_net(dataset, num_nets, args, device='cpu', base=False):
    nets = {}
    num_classes = dataset.num_classes
    norm_layer = None 

    for net_i in range(num_nets):
        net = None
        if args.model == 'resnet18':
            if base:
                net = resnet_cifar.ResNet18_cifar10(in_channels=args.in_channels, num_classes=num_classes, norm_layer=norm_layer, fan=args.fan, linit=args.linit, init=args.init)
            else:
                net = resnet_cifar.ResNet18_cifar10(in_channels=args.in_channels, num_classes=num_classes, norm_layer=norm_layer)
        elif args.model == 'resnet50':
            if base:
                net = resnet_cifar.ResNet50_cifar10(in_channels=args.in_channels, num_classes=num_classes, norm_layer=norm_layer, fan=args.fan, linit=args.linit, init=args.init)
            else:
                net = resnet_cifar.ResNet50_cifar10(in_channels=args.in_channels, num_classes=num_classes, norm_layer=norm_layer)
        elif args.model == 'resnet18_byot':
            net = multi_resnet18_kd(num_classes=num_classes, in_channels=args.in_channels)
        elif args.model == 'resnet50_byot':
            net = multi_resnet50_kd(num_classes=num_classes)
        elif args.model == 'mobilenet':
            if base:
                net = mobilenet_v2.MobileNetV2(num_classes=num_classes, in_channels=args.in_channels, norm_layer=norm_layer, fan=args.fan, linit=args.linit, last_fc=args.last_fc, no_init=args.no_init)
            else:
                net = mobilenet_v2.MobileNetV2(num_classes=num_classes, in_channels=args.in_channels, norm_layer=norm_layer, last_fc=args.last_fc, no_init=True)
        elif args.model == 'mobilenet_byot':
            net = mobilenet_v2.MobileNetV2BYOT(
                num_classes=num_classes,
                in_channels=args.in_channels,
                norm_layer=norm_layer,
                fan=args.fan,
                linit=args.linit,
                no_init=args.no_init if base else True,
            )
        elif args.model == 'simplecnn':
            net = simplecnn.SimpleCNN(num_classes=num_classes)
        else:
            raise ValueError(f"Unsupported model: {args.model}")

        if getattr(args, "group_norm", False):
            ng = getattr(args, "num_groups", 8)
            net = convert_bn_to_gn(net, num_groups=ng)

        net.to(device)
        nets[net_i] = net
    return nets

def init_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

def init_logger(args):
    os.makedirs(args.logdir, exist_ok=True)
    log_file_name = datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")
    if args.log_file_name is None:
        log_file_name = f'experiment-arguments_{log_file_name}'
    else:
        log_file_name = args.log_file_name

    log_path_prefix = os.path.join(args.logdir, log_file_name)
    os.makedirs(os.path.dirname(log_path_prefix), exist_ok=True)
    
    with open(log_path_prefix + '.json', 'w') as f:
        args_dict = vars(args)
        json.dump(args_dict, f, indent=4, ensure_ascii=False)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        filename=log_path_prefix + '.log',
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.INFO, filemode='w')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()            
    console_handler.setLevel(logging.INFO)                
    console_formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%m-%d %H:%M')                                                     
    console_handler.setFormatter(console_formatter)       
    logger.addHandler(console_handler)    
    
    init_seed(args)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return logger, log_file_name

@torch.no_grad()
def update_ema_model(ema_model, model, decay: float):
    """
    ema = decay * ema + (1-decay) * model
    - BN buffer 포함해서 같이 EMA 업데이트
    """
    ema_sd = ema_model.state_dict()
    sd = model.state_dict()

    for k, v in ema_sd.items():
        if k not in sd:
            continue
        src = sd[k]
        if not torch.is_floating_point(v):
            # e.g., num_batches_tracked 같은 int buffer는 그냥 복사
            ema_sd[k].copy_(src)
        else:
            v.mul_(decay).add_(src, alpha=1.0 - decay)

    ema_model.load_state_dict(ema_sd, strict=False)

def shuffle_clients(args):
    n_party_per_round = int(args.n_clients * args.sample_fraction)
    party_list = [i for i in range(args.n_clients)]
    party_list_rounds = []
    if n_party_per_round != args.n_clients:
        for i in range(args.round):
            party_list_rounds.append(random.sample(party_list, n_party_per_round))
    else:
        for i in range(args.round):
            party_list_rounds.append(party_list)
    return party_list_rounds

def compute_accuracy(model, dataloader, device):
    """
    BYOT 모델일 경우: Main Accuracy 외에 Ensemble Accuracy를 추가로 계산
    """
    was_training = model.training
    model.eval()

    correct_main = 0
    correct_ensemble = 0
    total = 0
    is_byot = False

    with torch.no_grad():
        for x, target in dataloader:
            x = x.to(device)
            target = target.to(dtype=torch.int64, device=device)
            out = model(x)

            if isinstance(out, tuple) and len(out) == 8:
                is_byot = True
                (output, m1, m2, m3, _, _, _, _) = out
                
                # 1. Main (Deepest) Prediction
                _, pred_main = torch.max(output.data, 1)
                
                # 2. Ensemble Prediction (Sum of Softmax)
                # 논문 방식: 단순히 Softmax 출력값들을 더함
                p_main = F.softmax(output, dim=1)
                p_m1 = F.softmax(m1, dim=1)
                p_m2 = F.softmax(m2, dim=1)
                p_m3 = F.softmax(m3, dim=1)
                
                ensemble_prob = p_main + p_m1 + p_m2 + p_m3
                _, pred_ensemble = torch.max(ensemble_prob.data, 1)

                correct_main += (pred_main == target).sum().item()
                correct_ensemble += (pred_ensemble == target).sum().item()
                
            else:
                # Standard Model
                if isinstance(out, tuple): 
                    logits = out[-1]
                else:
                    logits = out
                _, pred = torch.max(logits.data, 1)
                correct_main += (pred == target).sum().item()
            
            total += x.size(0)

    acc_main = correct_main / float(total) if total > 0 else 0.0
    
    if is_byot:
        acc_ensemble = correct_ensemble / float(total) if total > 0 else 0.0
        # Ensemble 결과 출력
        print(f"   [BYOT Eval] Main: {acc_main:.4f} | Ensemble: {acc_ensemble:.4f}", flush=True)
        # Main 대신 Ensemble 정확도를 반환하여 Best Model 기록에 반영하고 싶다면 아래 주석 해제
        # return acc_ensemble 
        return acc_main # 일단 Main 반환 (기존 비교 유지를 위해)

    if was_training: model.train()
    
    return acc_main

def avg_last_n(accuracy_list, n):
    if not accuracy_list:
        return None
    recent = accuracy_list[-n:] if len(accuracy_list) >= n else accuracy_list
    return sum(recent) / len(recent)
