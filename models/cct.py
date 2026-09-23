from torch.hub import load_state_dict_from_url
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils.transformers import TransformerClassifier
from .utils.tokenizer import Tokenizer
from .utils.helpers import pe_check, fc_check

# DXFL constructs models directly through ``init_net`` and does not require
# timm.  Keeping this decorator local prevents CCT support from adding a new
# dependency (or importing torchvision indirectly) to every existing run.
def register_model(model_factory):
    return model_factory

model_urls = {
    'cct_7_3x1_32':
        'https://shi-labs.com/projects/cct/checkpoints/pretrained/cct_7_3x1_32_cifar10_300epochs.pth',
    'cct_7_3x1_32_sine':
        'https://shi-labs.com/projects/cct/checkpoints/pretrained/cct_7_3x1_32_sine_cifar10_5000epochs.pth',
    'cct_7_3x1_32_c100':
        'https://shi-labs.com/projects/cct/checkpoints/pretrained/cct_7_3x1_32_cifar100_300epochs.pth',
    'cct_7_3x1_32_sine_c100':
        'https://shi-labs.com/projects/cct/checkpoints/pretrained/cct_7_3x1_32_sine_cifar100_5000epochs.pth',
    'cct_7_7x2_224_sine':
        'https://shi-labs.com/projects/cct/checkpoints/pretrained/cct_7_7x2_224_flowers102.pth',
    'cct_14_7x2_224':
        'https://shi-labs.com/projects/cct/checkpoints/pretrained/cct_14_7x2_224_imagenet.pth',
    'cct_14_7x2_384':
        'https://shi-labs.com/projects/cct/checkpoints/finetuned/cct_14_7x2_384_imagenet.pth',
    'cct_14_7x2_384_fl':
        'https://shi-labs.com/projects/cct/checkpoints/finetuned/cct_14_7x2_384_flowers102.pth',
}


class CCT(nn.Module):
    def __init__(self,
                 img_size=224,
                 embedding_dim=768,
                 n_input_channels=3,
                 n_conv_layers=1,
                 kernel_size=7,
                 stride=2,
                 padding=3,
                 pooling_kernel_size=3,
                 pooling_stride=2,
                 pooling_padding=1,
                 dropout=0.,
                 attention_dropout=0.1,
                 stochastic_depth=0.1,
                 num_layers=14,
                 num_heads=6,
                 mlp_ratio=4.0,
                 num_classes=1000,
                 positional_embedding='learnable',
                 *args, **kwargs):
        super(CCT, self).__init__()

        self.tokenizer = Tokenizer(n_input_channels=n_input_channels,
                                   n_output_channels=embedding_dim,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   pooling_kernel_size=pooling_kernel_size,
                                   pooling_stride=pooling_stride,
                                   pooling_padding=pooling_padding,
                                   n_conv_layers=n_conv_layers,
                                   )

        self.classifier = TransformerClassifier(
            sequence_length=self.tokenizer.sequence_length(n_channels=n_input_channels,
                                                           height=img_size,
                                                           width=img_size),
            embedding_dim=embedding_dim,
            seq_pool=True,
            dropout=dropout,
            attention_dropout=attention_dropout,
            stochastic_depth=stochastic_depth,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_classes=num_classes,
            positional_embedding=positional_embedding
        )

    def forward_features(self, x, return_intermediate_layers=()):
        tokens = self.tokenizer(x)
        return self.classifier.forward_features(
            tokens,
            return_intermediate_layers=return_intermediate_layers,
        )

    def forward(self, x):
        x = self.tokenizer(x)
        return self.classifier(x)


class CCTBranchHead(nn.Module):
    """Sequence-pooling classifier attached to an intermediate CCT block."""

    def __init__(self, embedding_dim, num_classes):
        super().__init__()
        self.norm = nn.LayerNorm(embedding_dim)
        self.attention_pool = nn.Linear(embedding_dim, 1)
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.apply(TransformerClassifier.init_weight)

    def forward(self, tokens):
        tokens = self.norm(tokens)
        attention = F.softmax(self.attention_pool(tokens), dim=1).transpose(-1, -2)
        features = torch.matmul(attention, tokens).squeeze(-2)
        return self.classifier(features), features


class CCTBYOT(CCT):
    """CCT-7 with three auxiliary BYOT exits.

    The default exits follow Transformer blocks 2, 4, and 5 (one-based).  They
    divide the seven-block backbone into early/middle/late observations while
    leaving blocks 6--7 exclusive to the final teacher path.  The return value
    intentionally matches the existing ResNet/MobileNet BYOT contract:

    ``(teacher, B1, B2, B3, teacher_feature, B1_feature, B2_feature, B3_feature)``.
    """

    def __init__(
        self,
        *args,
        branch_blocks=(2, 4, 5),
        embedding_dim=256,
        num_layers=7,
        num_classes=1000,
        **kwargs,
    ):
        if len(branch_blocks) != 3:
            raise ValueError(
                "The current DXFL BYOT loss expects exactly three CCT branches."
            )
        if tuple(sorted(set(branch_blocks))) != tuple(branch_blocks):
            raise ValueError("branch_blocks must be three unique increasing indices.")
        if branch_blocks[0] < 1 or branch_blocks[-1] >= num_layers:
            raise ValueError(
                "CCT branches must be after blocks in [1, num_layers - 1], "
                "leaving at least one final-only transformer block."
            )

        super().__init__(
            *args,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            **kwargs,
        )
        self.branch_blocks = tuple(int(index) for index in branch_blocks)
        self._branch_block_indices = tuple(index - 1 for index in branch_blocks)
        self.branch_heads = nn.ModuleList(
            [
                CCTBranchHead(embedding_dim, num_classes)
                for _ in self.branch_blocks
            ]
        )

    def forward_teacher(self, x):
        features, _ = self.forward_features(x)
        return features, self.classifier.fc(features)

    def forward_final_only(self, x):
        return self.forward_teacher(x)

    def forward(self, x, detach_branch_inputs=False):
        final_features, intermediate = self.forward_features(
            x,
            return_intermediate_layers=self._branch_block_indices,
        )
        teacher_logits = self.classifier.fc(final_features)

        branch_logits = []
        branch_features = []
        for head, block_index in zip(
            self.branch_heads, self._branch_block_indices
        ):
            branch_input = intermediate[block_index]
            if detach_branch_inputs:
                branch_input = branch_input.detach()
            logits, features = head(branch_input)
            branch_logits.append(logits)
            branch_features.append(features)

        return (
            teacher_logits,
            *branch_logits,
            final_features,
            *branch_features,
        )


def _cct(arch, pretrained, progress,
         num_layers, num_heads, mlp_ratio, embedding_dim,
         kernel_size=3, stride=None, padding=None,
         positional_embedding='learnable',
         *args, **kwargs):
    stride = stride if stride is not None else max(1, (kernel_size // 2) - 1)
    padding = padding if padding is not None else max(1, (kernel_size // 2))
    model = CCT(num_layers=num_layers,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                embedding_dim=embedding_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                *args, **kwargs)

    if pretrained:
        if arch in model_urls:
            state_dict = load_state_dict_from_url(model_urls[arch],
                                                  progress=progress)
            if positional_embedding == 'learnable':
                state_dict = pe_check(model, state_dict)
            elif positional_embedding == 'sine':
                state_dict['classifier.positional_emb'] = model.state_dict()['classifier.positional_emb']
            state_dict = fc_check(model, state_dict)
            model.load_state_dict(state_dict)
        else:
            raise RuntimeError(f'Variant {arch} does not yet have pretrained weights.')
    return model


@register_model
def cct_2(arch, pretrained, progress, *args, **kwargs):
    return _cct(arch, pretrained, progress, num_layers=2, num_heads=2, mlp_ratio=1, embedding_dim=128,
                *args, **kwargs)


@register_model
def cct_4(arch, pretrained, progress, *args, **kwargs):
    return _cct(arch, pretrained, progress, num_layers=4, num_heads=2, mlp_ratio=1, embedding_dim=128,
                *args, **kwargs)


@register_model
def cct_6(arch, pretrained, progress, *args, **kwargs):
    return _cct(arch, pretrained, progress, num_layers=6, num_heads=4, mlp_ratio=2, embedding_dim=256,
                *args, **kwargs)


@register_model
def cct_7(arch, pretrained, progress, *args, **kwargs):
    return _cct(arch, pretrained, progress, num_layers=7, num_heads=4, mlp_ratio=2, embedding_dim=256,
                *args, **kwargs)


@register_model
def cct_14(arch, pretrained, progress, *args, **kwargs):
    return _cct(arch, pretrained, progress, num_layers=14, num_heads=6, mlp_ratio=3, embedding_dim=384,
                *args, **kwargs)


@register_model
def cct_2_3x2_32(pretrained=False, progress=False,
                 img_size=32, positional_embedding='learnable', num_classes=10,
                 *args, **kwargs):
    return cct_2('cct_2_3x2_32', pretrained, progress,
                 kernel_size=3, n_conv_layers=2,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cct_2_3x2_32_sine(pretrained=False, progress=False,
                      img_size=32, positional_embedding='sine', num_classes=10,
                      *args, **kwargs):
    return cct_2('cct_2_3x2_32_sine', pretrained, progress,
                 kernel_size=3, n_conv_layers=2,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cct_4_3x2_32(pretrained=False, progress=False,
                 img_size=32, positional_embedding='learnable', num_classes=10,
                 *args, **kwargs):
    return cct_4('cct_4_3x2_32', pretrained, progress,
                 kernel_size=3, n_conv_layers=2,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cct_4_3x2_32_sine(pretrained=False, progress=False,
                      img_size=32, positional_embedding='sine', num_classes=10,
                      *args, **kwargs):
    return cct_4('cct_4_3x2_32_sine', pretrained, progress,
                 kernel_size=3, n_conv_layers=2,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cct_6_3x1_32(pretrained=False, progress=False,
                 img_size=32, positional_embedding='learnable', num_classes=10,
                 *args, **kwargs):
    return cct_6('cct_6_3x1_32', pretrained, progress,
                 kernel_size=3, n_conv_layers=1,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cct_6_3x1_32_sine(pretrained=False, progress=False,
                      img_size=32, positional_embedding='sine', num_classes=10,
                      *args, **kwargs):
    return cct_6('cct_6_3x1_32_sine', pretrained, progress,
                 kernel_size=3, n_conv_layers=1,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cct_6_3x2_32(pretrained=False, progress=False,
                 img_size=32, positional_embedding='learnable', num_classes=10,
                 *args, **kwargs):
    return cct_6('cct_6_3x2_32', pretrained, progress,
                 kernel_size=3, n_conv_layers=2,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cct_6_3x2_32_sine(pretrained=False, progress=False,
                      img_size=32, positional_embedding='sine', num_classes=10,
                      *args, **kwargs):
    return cct_6('cct_6_3x2_32_sine', pretrained, progress,
                 kernel_size=3, n_conv_layers=2,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cct_7_3x1_32(pretrained=False, progress=False,
                 img_size=32, positional_embedding='learnable', num_classes=10,
                 *args, **kwargs):
    return cct_7('cct_7_3x1_32', pretrained, progress,
                 kernel_size=3, n_conv_layers=1,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cct_7_3x1_32_sine(pretrained=False, progress=False,
                      img_size=32, positional_embedding='sine', num_classes=10,
                      *args, **kwargs):
    return cct_7('cct_7_3x1_32_sine', pretrained, progress,
                 kernel_size=3, n_conv_layers=1,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cct_7_3x1_32_c100(pretrained=False, progress=False,
                      img_size=32, positional_embedding='learnable', num_classes=100,
                      *args, **kwargs):
    return cct_7('cct_7_3x1_32_c100', pretrained, progress,
                 kernel_size=3, n_conv_layers=1,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cct_7_3x1_32_sine_c100(pretrained=False, progress=False,
                           img_size=32, positional_embedding='sine', num_classes=100,
                           *args, **kwargs):
    return cct_7('cct_7_3x1_32_sine_c100', pretrained, progress,
                 kernel_size=3, n_conv_layers=1,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cct_7_3x2_32(pretrained=False, progress=False,
                 img_size=32, positional_embedding='learnable', num_classes=10,
                 *args, **kwargs):
    return cct_7('cct_7_3x2_32', pretrained, progress,
                 kernel_size=3, n_conv_layers=2,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cct_7_3x2_32_byot(pretrained=False, progress=False,
                      img_size=32, positional_embedding='learnable',
                      num_classes=10, branch_blocks=(2, 4, 5),
                      *args, **kwargs):
    """CCT-7/3x2 with BYOT exits after blocks 2, 4, and 5."""
    if pretrained:
        raise ValueError(
            "A pretrained CCT-BYOT checkpoint is not provided; use pretrained=False."
        )
    return CCTBYOT(
        num_layers=7,
        num_heads=4,
        mlp_ratio=2,
        embedding_dim=256,
        kernel_size=3,
        n_conv_layers=2,
        img_size=img_size,
        positional_embedding=positional_embedding,
        num_classes=num_classes,
        branch_blocks=branch_blocks,
        *args,
        **kwargs,
    )


@register_model
def cct_7_3x2_32_sine(pretrained=False, progress=False,
                      img_size=32, positional_embedding='sine', num_classes=10,
                      *args, **kwargs):
    return cct_7('cct_7_3x2_32_sine', pretrained, progress,
                 kernel_size=3, n_conv_layers=2,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cct_7_7x2_224(pretrained=False, progress=False,
                  img_size=224, positional_embedding='learnable', num_classes=102,
                  *args, **kwargs):
    return cct_7('cct_7_7x2_224', pretrained, progress,
                 kernel_size=7, n_conv_layers=2,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cct_7_7x2_224_sine(pretrained=False, progress=False,
                       img_size=224, positional_embedding='sine', num_classes=102,
                       *args, **kwargs):
    return cct_7('cct_7_7x2_224_sine', pretrained, progress,
                 kernel_size=7, n_conv_layers=2,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cct_14_7x2_224(pretrained=False, progress=False,
                   img_size=224, positional_embedding='learnable', num_classes=1000,
                   *args, **kwargs):
    return cct_14('cct_14_7x2_224', pretrained, progress,
                  kernel_size=7, n_conv_layers=2,
                  img_size=img_size, positional_embedding=positional_embedding,
                  num_classes=num_classes,
                  *args, **kwargs)


@register_model
def cct_14_7x2_384(pretrained=False, progress=False,
                   img_size=384, positional_embedding='learnable', num_classes=1000,
                   *args, **kwargs):
    return cct_14('cct_14_7x2_384', pretrained, progress,
                  kernel_size=7, n_conv_layers=2,
                  img_size=img_size, positional_embedding=positional_embedding,
                  num_classes=num_classes,
                  *args, **kwargs)


@register_model
def cct_14_7x2_384_fl(pretrained=False, progress=False,
                      img_size=384, positional_embedding='learnable', num_classes=102,
                      *args, **kwargs):
    return cct_14('cct_14_7x2_384_fl', pretrained, progress,
                  kernel_size=7, n_conv_layers=2,
                  img_size=img_size, positional_embedding=positional_embedding,
                  num_classes=num_classes,
                  *args, **kwargs)
