'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np

# resnet_cifar_repr.py 파일 하단에 추가

class LoRAConv2d(nn.Module):
    """ FLoCoRA 논문을 기반으로 nn.Conv2d 레이어에 LoRA를 적용하는 wrapper 클래스 """
    def __init__(self, original_layer, rank, alpha):
        super().__init__()
        self.original_layer = original_layer
        
        # 원본 레이어의 파라미터는 동결
        for param in self.original_layer.parameters():
            param.requires_grad = False
            
        in_channels = original_layer.in_channels
        out_channels = original_layer.out_channels
        kernel_size = original_layer.kernel_size
        stride = original_layer.stride
        padding = original_layer.padding

        self.rank = rank
        self.alpha = alpha
        
        # LoRA 어댑터 행렬 A와 B (논문에서는 Conv 레이어로 구현)
        # B: in_channels -> rank
        # A: rank -> out_channels
        self.lora_B = nn.Conv2d(in_channels, rank, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.lora_A = nn.Conv2d(rank, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        
        # 초기화: A는 Kaiming, B는 0으로
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        # 원본 레이어 출력 + LoRA 어댑터 출력
        original_output = self.original_layer(x)
        lora_output = self.lora_A(self.lora_B(x))
        
        # 스케일링 적용 (논문 수식 참조)
        return original_output + (self.alpha / self.rank) * lora_output

def apply_flocora_to_model(model, args):
    """
    모든 Conv2d를 LoRAConv2d로 교체 + (중요) 정규화층/최종 분류기 언프리즈 보장
    """
    last_linear = None  # 마지막 nn.Linear(= 최종 FC) 추적

    # 1) 재귀적으로 Conv2d → LoRAConv2d 교체
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            apply_flocora_to_model(module, args)  # 하위 모듈 먼저 처리

        # 최종 FC를 찾기 위해 Linear 추적
        if isinstance(module, nn.Linear):
            last_linear = module  # 마지막에 본 Linear가 최종 FC일 가능성 큼

        # Conv2d를 LoRA로 교체
        if isinstance(module, nn.Conv2d):
            lora_layer = LoRAConv2d(module, rank=args.lora_r, alpha=args.lora_alpha)
            setattr(model, name, lora_layer)

    # 2) (중요) 정규화층과 최종 분류기 언프리즈 보장
    #    - BN/GN 등 정규화층: 로컬 통계 적응 위해 항상 학습 가능
    #    - 최종 FC(teacher head): 워밍업에서 teacher만 학습하므로 무조건 학습 가능
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
            for p in m.parameters():
                p.requires_grad = True

    # 최종 FC가 실제로 있었다면 언프리즈
    if last_linear is not None:
        for p in last_linear.parameters():
            p.requires_grad = True

    # 3) (선택) 이름 기반으로도 안전망 추가 — 모델 네이밍에 맞춰 키워드 보강 가능
    for n, p in model.named_parameters():
        lname = n.lower()
        if ("classifier" in lname) or (".fc." in lname) or lname.endswith(".fc.weight") or lname.endswith(".fc.bias"):
            p.requires_grad = True

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetCifar10(nn.Module):

    def __init__(self, block, layers, in_channels=3, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, fan='fan_in', linit=False, init='normal'):
        super(ResNetCifar10, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.in_channels = in_channels
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(self.in_channels, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        print(f'{init} init')
        
        if init == 'normal':
            if linit:
                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.kaiming_normal_(m.weight, mode=fan, nonlinearity='relu')
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
                    if isinstance(m, nn.Linear):
                        nn.init.kaiming_normal_(m.weight, mode=fan,nonlinearity='linear')
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)

                    if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)

            else:
                for m in self.modules():
                    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                        nn.init.kaiming_normal_(m.weight, mode=fan, nonlinearity='relu')
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)

                    if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)
        elif init == 'kaiming_uniform':
            if linit:
                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.kaiming_uniform_(m.weight, mode=fan, nonlinearity='relu')
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
                    if isinstance(m, nn.Linear):
                        nn.init.kaiming_uniform_(m.weight, mode=fan,nonlinearity='linear')
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)

                    if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)

            else:
                for m in self.modules():
                    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                        nn.init.kaiming_uniform_(m.weight, mode=fan, nonlinearity='relu')
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)

                    if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)
        elif init == 'orthogonal':
            # orthogonal_init(net ,args)
            print('orthogonal init')
            for m in self.modules():
                if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                    torch.nn.init.orthogonal_(m.weight)
                    if m.bias is not None:
                        torch.nn.init.constant_(m.bias, 0)
        elif init == 'kaiming_orthogonal':
            # orthogonal_init(net ,args)
            print('kaiming_orthogonal init')
            for m in self.modules():
                if isinstance(m, torch.nn.Conv2d):
                    torch.nn.init.orthogonal_(m.weight)
                    with torch.no_grad():
                        rows = m.weight.size(0)  # out_channels
                        cols = m.weight.numel() // rows  # in_channels * kernel_height * kernel_width 
                        m.weight.data = m.weight.data * nn.init.calculate_gain('relu') * np.sqrt( max(rows, cols) / nn.init._calculate_correct_fan(m.weight, fan))
                    if m.bias is not None:
                        torch.nn.init.constant_(m.bias, 0)
                elif isinstance(m, torch.nn.Linear):
                    torch.nn.init.orthogonal_(m.weight)
                    with torch.no_grad():
                        rows = m.weight.size(0)  # out_channels
                        cols = m.weight.numel() // rows  # in_channels * kernel_height * kernel_width 
                        if linit:
                            m.weight.data = m.weight.data * nn.init.calculate_gain('linear') * np.sqrt( max(rows, cols) / nn.init._calculate_correct_fan(m.weight, fan))
                        else:
                            m.weight.data = m.weight.data * nn.init.calculate_gain('relu') * np.sqrt( max(rows, cols) / nn.init._calculate_correct_fan(m.weight, fan))
                    if m.bias is not None:
                        torch.nn.init.constant_(m.bias, 0)
        else:
            if linit:
                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.kaiming_normal_(m.weight, mode=fan, nonlinearity='relu')
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
                    if isinstance(m, nn.Linear):
                        nn.init.kaiming_normal_(m.weight, mode=fan,nonlinearity='linear')
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)

                    if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, fa=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        # pre_fa, post_fa = self._count_fa()
        # layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
        #                     self.base_width, previous_dilation, norm_layer, pre_fa=self.pre_fa, post_fa=self.post_fa))
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                            base_width=self.base_width, dilation=self.dilation,
                            norm_layer=norm_layer))

        return nn.Sequential(*layers)
    
    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        features = torch.flatten(x, 1)
        y = self.fc(features)

        return features, y

    def forward(self, x):
        return self._forward_impl(x)


def ResNet18_cifar10(**kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return ResNetCifar10(BasicBlock, [2, 2, 2, 2], **kwargs)



def ResNet50_cifar10(**kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return ResNetCifar10(Bottleneck, [3, 4, 6, 3], **kwargs)