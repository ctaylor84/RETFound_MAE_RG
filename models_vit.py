# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm.models.vision_transformer


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def efficientnetv2_m(**kwargs):
    model = timm.create_model("efficientnetv2_rw_m", pretrained=True,
                              num_classes=kwargs["num_classes"])
    return model

def resnet50(**kwargs):
    model = timm.create_model("resnet50", pretrained=True,
                              num_classes=kwargs["num_classes"])
    return model



class StackConverter(nn.Module):
    def __init__(self):
        super(StackConverter, self).__init__()

        self.block_1 = nn.Sequential(
            nn.Conv3d(3, 8, (1, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(True),
            nn.Conv3d(8, 8, (1, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(True),
            nn.Conv3d(8, 8, (3, 1, 1)),
        )
        self.iden_1 = nn.Conv3d(3, 8, (3, 1, 1))

        self.block_2 = nn.Sequential(
            nn.Conv3d(8, 8, (1, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(True),
            nn.Conv3d(8, 8, (1, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(True),
            nn.Conv3d(8, 3, (3, 1, 1)),
        )
        self.iden_2 = nn.Conv3d(8, 3, (3, 1, 1))
    
    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        identity = self.iden_1(x)
        x = self.block_1(x)
        x += identity
        x = F.relu(x, True)

        identity = self.iden_2(x)
        x = self.block_2(x)
        x += identity
        x = F.relu(x, True)
        x = x.squeeze(2)
        return x

class VisionTransformerStack(VisionTransformer):
    def __init__(self, **kwargs):
        super(VisionTransformerStack, self).__init__(**kwargs)
        self.stack = StackConverter()

    def forward_features(self, x):
        x = self.stack(x)
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

def vit_large_patch16_stack(**kwargs):
    model = VisionTransformerStack(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

class Efficientnetv2mStack(nn.Module):
    def __init__(self, **kwargs):
        super(Efficientnetv2mStack, self).__init__()
        self.stack = StackConverter()
        self.model = timm.create_model("efficientnetv2_rw_m", pretrained=True,
                                       num_classes=kwargs["num_classes"])

    def forward(self, x):
        x = self.stack(x)
        x = self.model(x)
        return x

def efficientnetv2_m_stack(**kwargs):
    model = Efficientnetv2mStack(**kwargs)
    return model

class Resnet50Stack(nn.Module):
    def __init__(self, **kwargs):
        super(Resnet50Stack, self).__init__()
        self.stack = StackConverter()
        self.model = timm.create_model("resnet50", pretrained=True,
                                       num_classes=kwargs["num_classes"])

    def forward(self, x):
        x = self.stack(x)
        x = self.model(x)
        return x

def resnet50_stack(**kwargs):
    model = Resnet50Stack(**kwargs)
    return model
