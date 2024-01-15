"""
Mostly copy-paste from timm library.
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
import math
from functools import partial

import torch
import torch.nn as nn




def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=3, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale


        # diagnol filled +1
        # a = torch.zeros(286, 286).to('cuda')
        # a = a.fill_diagonal_(1)
        # for k in range(attn.size(0)):
        #     for i in range(attn.size(1)):
        #         attn[k][i] = attn[k][i] + a




        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))

        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if return_attention:
            return x, attn

        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=284, patch_size=1, in_chans=116, embed_dim=90):
        super().__init__()
        num_patches = (img_size // patch_size)   # 하나의 이미지 안 patch 개수
        self.img_size = img_size  # 284
        self.patch_size = patch_size  # 1
        self.num_patches = num_patches  # 284

        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, H, W = x.shape # (64, 284, 116)
        x = x.transpose(1,2)  # (64, 116, 284)
        x = self.proj(x).transpose(1, 2)  # (64, 80, 284) -> (64, 284, 80)
        return x  # (64, 284, 80) patch sequence = 284 patch 1개당 80 dimension


class VisionTransformer(nn.Module):   # img_size [284]
    """ Vision Transformer """

    def __init__(self, img_size=[224], patch_size=1, in_chans=116, num_classes=0, embed_dim=90, depth=12,
                 num_heads=3, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm, headData=True, headClass=True, **kwargs):
        super().__init__()

        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches  # 284 // 1 -> 284

        self.cls_token1 = nn.Parameter(torch.zeros(1, 1, embed_dim))  # torch.Size([1, 1, 116])
        self.cls_token2 = nn.Parameter(torch.zeros(1, 1, embed_dim))  # torch.Size([1, 1, 116])
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, embed_dim))  # torch.Size([1, 286, 116])
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)



        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token1, std=.02)
        nn.init.trunc_normal_(self.cls_token2, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # from [https://github.com/facebookresearch/dino/issues/8]
    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 2
        N = self.pos_embed.shape[1] - 2
        if npatch == N :
            return self.pos_embed


        class_pos_embed = self.pos_embed[:, 0:2]
        patch_pos_embed = self.pos_embed[:, 2:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, w, h = x.shape  # (64, 284, 116)
        x = self.patch_embed(x)  # patch linear embedding  (64, 284, 80)

        # add the [CLS] token to the embed patch tokens
        cls_tokens1 = self.cls_token1.expand(B, -1, -1)
        cls_tokens2 = self.cls_token2.expand(B, -1, -1)
        x = torch.cat((cls_tokens1, cls_tokens2, x), dim=1)


        x = x + self.interpolate_pos_encoding(x, w, h)   # add positional encoding to each token

        return self.pos_drop(x)

    def forward(self, x, classify=False):

        x = self.prepare_tokens(x)   # torch.Size([64, 286, 80])  embed 1단계

        for i, blk in enumerate(self.blocks):  # block 0~11  transformer encoder
            x = blk(x)                         # torch.Size([64, 18, 768])

        x = self.norm(x)       # torch.Size([64, 286, 80])  partial(nn.LayerNorm, eps=1e-6)
        return x


    def get_last_features_attentions(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):

            if i == (len(self.blocks) - 1):
                x, attn = blk(x, return_attention=True)
                return self.norm(x), attn

            else:
                x = blk(x)

    def get_all_features_attentions(self, x):
        x = self.prepare_tokens(x)

        attn = []
        for i, blk in enumerate(self.blocks):
            x, attn_ = blk(x, return_attention=True)
            attn.append(attn_)

        return attn









def vit_tiny(patch_size=1, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=90, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )

    return model








########### DECODER ##########
class RECHead(nn.Module):
    def __init__(self, in_dim, in_chans=116, patch_size=1):
        super().__init__()

        layers = [nn.Linear(in_dim, in_dim)]
        layers.append(nn.GELU())
        layers.append(nn.Linear(in_dim, in_dim))
        layers.append(nn.GELU())

        self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)

        self.convTrans = nn.ConvTranspose1d(in_dim, in_chans, kernel_size=patch_size,
                                            stride=patch_size)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)  # torch.Size([64, 284, 80])

        x_rec = x.transpose(1, 2)  # torch.Size([64, 80, 284])
        x_rec = self.convTrans(x_rec)  # torch.Size([64, 116, 284])
        x_rec = x_rec.permute(0, 2, 1)  # torch.Size([64, 284, 116])

        return x_rec


