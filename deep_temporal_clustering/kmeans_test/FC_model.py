import torch.nn as nn
import torch
from functools import partial




class TAE_encoder(nn.Module):
    """
    Class for temporal autoencoder encoder.
    filter_1 : filter size of the first convolution layer
    filter_lstm : hidden size of the lstm.
    pooling : pooling number for maxpooling.
    """

    def __init__(self, filter_1, filter_2, filter_3, filter_lstm):
        super(TAE_encoder, self).__init__()

        self.hidden_lstm_1 = filter_lstm[0]
        self.hidden_lstm_2 = filter_lstm[1]


        ## CNN PART
        ### output shape (batch_size, 7 , n_hidden = 284)
        self.conv_layer_1 = nn.Sequential(
            nn.Conv1d(
                in_channels=6670,
                out_channels=filter_1,
                kernel_size=1,
                stride=1
            ),
            nn.LeakyReLU(),
        )
        self.conv_layer_2 = nn.Sequential(
            nn.Conv1d(
                in_channels=filter_1,
                out_channels=filter_2,
                kernel_size=1,
                stride=1
            ),
            nn.LeakyReLU(),
        )
        self.conv_layer_3 = nn.Sequential(
            nn.Conv1d(
                in_channels=filter_2,
                out_channels=filter_3,
                kernel_size=1,
                stride=1
            ),
            nn.LeakyReLU(),
        )

        ## LSTM PART
        ### output shape (batch_size , n_hidden = 284 , 50)
        self.lstm_1 = nn.LSTM(
            input_size= filter_3,
            hidden_size=self.hidden_lstm_1,
            batch_first=True

        )
        self.act_lstm_1 = nn.Tanh()


        ### output shape (batch_size , n_hidden = 284 , 10)
        self.lstm_2 = nn.LSTM(
            input_size=self.hidden_lstm_1,
            hidden_size=self.hidden_lstm_2,
            batch_first=True

        )
        self.act_lstm_2 = nn.Tanh()


    def forward(self, x):  # x : (1, 284, 116)

        ## encoder
        x = x.transpose(1,2) # (1, 6670, 6)
        x = self.conv_layer_1(x) # (1, 1024, 6)
        x = self.conv_layer_2(x) # (1, 512, 6)
        x = self.conv_layer_3(x)  # (1, 256, 6)
        out_cnn = x.permute(0, 2, 1) # (1, 6, 256)

        out_lstm1, _ = self.lstm_1(out_cnn) # (1, 6, 128)
        out_lstm1_act = self.act_lstm_1(out_lstm1)


        features, _ = self.lstm_2(out_lstm1_act) # (1, 6, 64)
        out_lstm2_act = self.act_lstm_2(features)



        return out_lstm2_act


class TAE_decoder(nn.Module):
    """
    Class for temporal autoencoder decoder.
    filter_1 : filter size of the first convolution layer
    filter_lstm : hidden size of the lstm.
    """

    def __init__(self, filter_1, filter_2, filter_lstm):
        super(TAE_decoder, self).__init__()

        self.hidden_lstm_1 = filter_lstm[0]
        self.hidden_lstm_2 = filter_lstm[1]

        # upsample
        self.deconv_layer_1 = nn.ConvTranspose1d(
            in_channels=self.hidden_lstm_2 ,
            out_channels=filter_2,
            kernel_size=1,
            stride=1,
        )
        self.deconv_layer_2 = nn.ConvTranspose1d(
            in_channels=filter_2,
            out_channels=filter_1,
            kernel_size=1,
            stride=1,
        )
        self.deconv_layer_3 = nn.ConvTranspose1d(
            in_channels=filter_1,
            out_channels=6670,
            kernel_size=1,
            stride=1,
        )

    def forward(self, features):

        ## decoder
        features = features.transpose(1, 2)   # (1, 64, 6)
        features = self.deconv_layer_1(features) # (1, 512, 6)
        features = self.deconv_layer_2(features) # (1, 1024, 6)
        features = self.deconv_layer_3(features) # (1, 6670, 6)
        out_deconv = features.permute(0,2,1) # (1, 6, 6670)


        return out_deconv




class TAE(nn.Module):
    """
    Class for temporal autoencoder.
    filter_1 : filter size of the first convolution layer
    filter_lstm : hidden size of the lstm.
    """

    def __init__(self, args, filter_1=1024, filter_2=512, filter_3=256, filter_lstm=[128, 64]):  # 내 모델에 사용될 구성품을 정의 및 초기화하는 메서드
        super(TAE, self).__init__()

        self.filter_1 = filter_1
        self.filter_2 = filter_2
        self.filter_3 = filter_3
        self.filter_lstm = filter_lstm

        self.tae_encoder = TAE_encoder(
            filter_1=self.filter_1,
            filter_2=self.filter_2,
            filter_3=self.filter_3,
            filter_lstm=self.filter_lstm,
        )

        self.tae_decoder = TAE_decoder(
            filter_1=self.filter_1,
            filter_2=self.filter_2,
            filter_lstm=self.filter_lstm
        )
        self.device = device = torch.device('cuda')
        self.mask_token = torch.zeros((1, 1, 6670), device=device)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)


        return x_masked, mask, ids_restore

    def masking_tokens(self, x, mask_ratio):

        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x[:, :, :], mask_tokens], dim=1)  # no cls token
        q = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))

        return q



    def forward(self, x):        #  init에서 정의된 구성품들을 연결하는 메서드

        x_ = self.masking_tokens(x, 0.15)
        features = self.tae_encoder(x_)
        out_deconv = self.tae_decoder(features)

        return features, out_deconv   # features는 clustering을 위해 encoder의 output을 사용










class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()


        self.fc1 = nn.Linear(in_features=6670, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=2)
        # self.fc4 = nn.Linear(in_features=64, out_features=2)



    def forward(self, x):  # x : (1, 284, 116)

        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        # x = nn.functional.softmax(self.fc3(x))
        x = self.fc3(x)

        return x






























###########################################################################################################
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
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
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

        attn = (q @ k.transpose(-2, -1)) * self.scale #(1)


        # diagnol filled +1
        # a = torch.zeros(286, 286).to('cuda')
        # a = a.fill_diagonal_(1)
        # for k in range(attn.size(0)):
        #     for i in range(attn.size(1)):
        #         attn[k][i] = attn[k][i] + a




        attn = attn.softmax(dim=-1) #(2)
        attn = self.attn_drop(attn) #(3)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_hidden_dim, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm3 = norm_layer(dim)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x)) # x(16, 284, 64), y(16, 284, 64), attn(16, 2, 284, 284)

        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        x = self.norm3(x)  ### modified  partial(nn.LayerNorm, eps=1e-6)

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
        x = x.transpose(1,2)  # (16, 116, 284)
        x = self.proj(x).transpose(1, 2)  # (64, 80, 284) -> (64, 284, 80)
        return x  # (64, 284, 80) patch sequence = 284 patch 1개당 80 dimension






############# ENCODER ##############
class VisionTransformer(nn.Module):   # img_size [284]
    """ Vision Transformer """

    def __init__(self, img_size=224, patch_size=1, in_chans=116, embed_dim=90, depth=12,
                 num_heads=2, mlp_hidden_dim=8, qkv_bias=False, norm_layer=nn.LayerNorm,
                 qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1):
        super().__init__()

        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches  # 284 // 1 -> 284


        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches , embed_dim))  # torch.Size([1, 286, 116])
        self.pos_drop = nn.Dropout(p=drop_rate)


        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_hidden_dim=mlp_hidden_dim, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,  norm_layer=norm_layer
            ) for i in range(depth)
        ])

        self.device = device = torch.device('cuda')
        self.mask_token = torch.zeros((1, 1, 6670), device=device)

        nn.init.trunc_normal_(self.pos_embed, std=.02)


    def masking_tokens(self, x, mask_ratio):

        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x[:, :, :], mask_tokens], dim=1)  # no cls token
        q = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))

        return q

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)


        return x_masked, mask, ids_restore




    # from [https://github.com/facebookresearch/dino/issues/8]
    def interpolate_pos_encoding(self, x, w, h):

        return self.pos_embed


    def prepare_tokens(self, x):
        B, w, h = x.shape  # (16, 284, 116)
        x = self.patch_embed(x)  # patch linear embedding  (16, 284, 64)

        x = x + self.interpolate_pos_encoding(x, w, h)   # add positional encoding to each token (16, 284, 64) + (1, 284, 64)

        return self.pos_drop(x)

    def forward(self, x):    # encoding layer

        x_ = self.masking_tokens(x, 0.15)

        x = self.prepare_tokens(x_)   # torch.Size([16, 286, 64])  embed 1단계

        for i, blk in enumerate(self.blocks):  # block 1  transformer encoder
            x = blk(x)                         # torch.Size([64, 18, 768])

        return x













########### DECODER ##########
class RECHead(nn.Module):
    def __init__(self, patch_size, embed_dim, in_chans, num_heads, mlp_hidden_dim, qkv_bias=False, norm_layer=nn.LayerNorm, depth=1):
        super().__init__()


        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_hidden_dim=mlp_hidden_dim, qkv_bias=qkv_bias, qk_scale=None,
                drop=0., attn_drop=0., drop_path=0., norm_layer=norm_layer
            ) for i in range(depth)
        ])


        self.convTrans = nn.ConvTranspose1d(embed_dim, in_chans, kernel_size=patch_size, stride=patch_size)



    def forward(self, x):

        for i, blk in enumerate(self.blocks):  # block 1  transformer encoder
            x = blk(x)  # (16, 284, 64)

        x_rec = x.transpose(1, 2)  # (16, 64, 284)
        x_rec = self.convTrans(x_rec) # (16, 116, 284)
        x_rec = x_rec.permute(0, 2, 1) # (16, 284, 116)


        return x_rec






class FullpiplineSiT(nn.Module):

    def __init__(self, args):
        super(FullpiplineSiT, self).__init__()



        # create full model
        #### ENCODER ####
        self.backbone =  VisionTransformer( img_size= 6, patch_size=1,
                                            embed_dim=1024, in_chans=6670, num_heads=2, mlp_hidden_dim=128,
                                            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                            depth=1
                                        )


        #### DECODER ####
        self.rec_head = RECHead(patch_size=1,
                                embed_dim=1024, in_chans=6670, num_heads=2, mlp_hidden_dim=128,
                                qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                depth=1
                            )



    def forward(self, im):


        #### ENCODER ####
        encoded_out = self.backbone(im) # torch.Size([16, 284, 116]) -> torch.Size([16, 286, 64])


        #### DECODER ####
        recons_imgs = self.rec_head(encoded_out)  # torch.Size([16, 286, 64]) -> torch.Size([16, 286, 116])




        return encoded_out, recons_imgs












