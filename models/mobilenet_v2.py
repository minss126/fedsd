"""mobilenetv2 in pytorch



[1] Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen

    MobileNetV2: Inverted Residuals and Linear Bottlenecks
    https://arxiv.org/abs/1801.04381
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearBottleNeck(nn.Module):

    def __init__(self, in_channels, out_channels, stride, norm_layer, t=6):
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * t, 1, bias=False),
            norm_layer(in_channels * t),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels * t, in_channels * t, 3, stride=stride, padding=1, groups=in_channels * t, bias=False),
            norm_layer(in_channels * t),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels * t, out_channels, 1, bias=False),
            norm_layer(out_channels)
        )

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):

        residual = self.residual(x)

        if self.stride == 1 and self.in_channels == self.out_channels:
            residual += x

        return residual

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=100, in_channels=3, norm_layer=None, fan='fan_in', linit=False, last_fc=False, no_init=False):
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.pre = nn.Sequential(
            nn.Conv2d(in_channels, 32, 1, padding=1, bias=False),
            norm_layer(32),
            nn.ReLU6(inplace=True)
        )

        self.stage1 = LinearBottleNeck(32, 16, 1, norm_layer, 1)
        self.stage2 = self._make_stage(2, 16, 24, 2, norm_layer, 6)
        self.stage3 = self._make_stage(3, 24, 32, 2, norm_layer, 6)
        self.stage4 = self._make_stage(4, 32, 64, 2, norm_layer, 6)
        self.stage5 = self._make_stage(3, 64, 96, 1, norm_layer, 6)
        self.stage6 = self._make_stage(3, 96, 160, 1, norm_layer, 6)
        self.stage7 = LinearBottleNeck(160, 320, 1, norm_layer, 6)

        self.conv1 = nn.Sequential(
            nn.Conv2d(320, 1280, 1, bias=False),
            norm_layer(1280),
            nn.ReLU6(inplace=True)
        )

        if last_fc:
            self.classifier = nn.Linear(1280, num_classes)
        else:
            self.classifier = nn.Conv2d(1280, num_classes, 1)

        if not no_init:
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

    def forward(self, x):
        x = self.pre(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.conv1(x)
        features = F.adaptive_avg_pool2d(x, 1)
        if isinstance(self.classifier, nn.Linear):
            features = features.view(features.size(0), -1)
        y = self.classifier(features)
        y = y.view(y.size(0), -1)

        return features, y

    def _make_stage(self, repeat, in_channels, out_channels, stride, norm_layer, t):

        layers = []
        layers.append(LinearBottleNeck(in_channels, out_channels, stride, norm_layer, t))

        while repeat - 1:
            layers.append(LinearBottleNeck(out_channels, out_channels, 1, norm_layer, t))
            repeat -= 1

        return nn.Sequential(*layers)

class MobileNetV2BYOT(MobileNetV2):
    """MobileNetV2 with three auxiliary BYOT branches.

    The output contract matches the existing ResNet BYOT model:
    (teacher_logits, b1_logits, b2_logits, b3_logits,
     teacher_feature, b1_feature, b2_feature, b3_feature).
    """

    def __init__(self, num_classes=100, in_channels=3, norm_layer=None,
                 fan='fan_in', linit=False, no_init=False):
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            norm_layer=norm_layer,
            fan=fan,
            linit=linit,
            last_fc=True,
            no_init=no_init,
        )
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.branch1 = self._make_branch(24, 1280, num_classes, norm_layer)
        self.branch2 = self._make_branch(64, 1280, num_classes, norm_layer)
        self.branch3 = self._make_branch(160, 1280, num_classes, norm_layer)

    @staticmethod
    def _make_branch(in_channels, feature_dim, num_classes, norm_layer):
        return nn.ModuleDict({
            'adapter': nn.Sequential(
                nn.Conv2d(in_channels, feature_dim, kernel_size=1, bias=False),
                norm_layer(feature_dim),
                nn.ReLU6(inplace=True),
                nn.AdaptiveAvgPool2d(1),
            ),
            'classifier': nn.Linear(feature_dim, num_classes),
        })

    @staticmethod
    def _branch_forward(branch, x):
        feature = branch['adapter'](x)
        logits = branch['classifier'](torch.flatten(feature, 1))
        return logits, feature

    def forward(self, x):
        x = self.pre(x)
        x = self.stage1(x)

        x = self.stage2(x)
        branch1_logits, branch1_feature = self._branch_forward(self.branch1, x)

        x = self.stage3(x)
        x = self.stage4(x)
        branch2_logits, branch2_feature = self._branch_forward(self.branch2, x)

        x = self.stage5(x)
        x = self.stage6(x)
        branch3_logits, branch3_feature = self._branch_forward(self.branch3, x)

        x = self.stage7(x)
        x = self.conv1(x)
        teacher_feature = F.adaptive_avg_pool2d(x, 1)
        teacher_logits = self.classifier(torch.flatten(teacher_feature, 1))

        return (
            teacher_logits,
            branch1_logits,
            branch2_logits,
            branch3_logits,
            teacher_feature,
            branch1_feature,
            branch2_feature,
            branch3_feature,
        )

def mobilenetv2():
    return MobileNetV2()

def mobilenetv2_byot(num_classes=100, in_channels=3):
    return MobileNetV2BYOT(num_classes=num_classes, in_channels=in_channels)
