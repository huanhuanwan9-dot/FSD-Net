# dpt.py
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
import torch.fft as fft
from einops import rearrange
# 确保安装了: pip install torch-dct pytorch_wavelets einops timm
import torch_dct as DCT
from blocks import FeatureFusionBlock, _make_scratch
from timm.models.layers import DropPath
from pytorch_wavelets import DWTForward


# ===================== 基础工具与 LayerNorm =====================
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


# ===================== 官方原版 PFESA =====================
class PFESA(nn.Module):
    def __init__(self, base_ratio=0.1):
        super(PFESA, self).__init__()
        self.activation = nn.Sigmoid()
        self.base_ratio = base_ratio
        self.eps = 1e-5

    def _edge_attention(self, x):
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        x_var = x.var(dim=[2, 3], keepdim=True)
        return x_minus_mu_square / (x_var + self.eps)

    def _structure_attention(self, x):
        energy_low = torch.pow(x, 2)
        energy_mu = torch.mean(energy_low, dim=[2, 3], keepdim=True)
        energy_var = torch.var(energy_low, dim=[2, 3], keepdim=True)
        y = (energy_low - energy_mu) / (energy_var + self.eps)
        return self.activation(y)

    def forward(self, x):
        origin_dtype = x.dtype
        x = x.to(torch.float32)
        b, c, h, w = x.size()

        x_freq = fft.fftn(x, dim=(-2, -1))
        x_freq = fft.fftshift(x_freq, dim=(-2, -1))

        low_freq_mask = self._create_low_freq_mask(h, w, device=x_freq.device)
        high_freq_mask = 1 - low_freq_mask

        low_freq = torch.abs(fft.ifftn(x_freq * low_freq_mask, dim=(-2, -1)))
        high_freq = torch.abs(fft.ifftn(x_freq * high_freq_mask, dim=(-2, -1)))

        out_att = self._structure_attention(low_freq) + self._edge_attention(high_freq)
        return (self.activation(out_att) * x).to(origin_dtype)

    def _create_low_freq_mask(self, h, w, device='cpu'):
        mask_ratio = self.base_ratio * min(h, w) / max(h, w)
        y = torch.linspace(-1, 1, h, device=device)
        x = torch.linspace(-1, 1, w, device=device)
        Y, X = torch.meshgrid(y, x, indexing='ij')
        return torch.exp(-(Y ** 2 + X ** 2) / (2 * mask_ratio ** 2))


# ===================== FCM 组件 (恢复标准卷积提精度) =====================
class ConvolutionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, groups=1):
        super().__init__()
        padding = kernel_size // 2 if padding is None else padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.SiLU()

    def forward(self, x):
        return self.activation(self.batch_norm(self.conv(x)))


class SpatialAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.batch_norm = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.batch_norm(self.conv(x)))


class ChannelAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.global_pooling(self.depthwise_conv(x)))


class FCM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.main_channels = channels - channels // 4
        self.sub_channels = channels // 4
        # 恢复标准3x3卷积，确保通道间特征有效融合
        self.main_branch_conv1 = ConvolutionLayer(self.main_channels, self.main_channels, kernel_size=3)
        self.main_branch_conv2 = ConvolutionLayer(self.main_channels, self.main_channels, kernel_size=3)
        self.main_branch_conv3 = ConvolutionLayer(self.main_channels, channels, kernel_size=1)
        self.sub_branch_conv = ConvolutionLayer(self.sub_channels, channels, kernel_size=1)
        self.spatial_attention = SpatialAttention(channels)
        self.channel_attention = ChannelAttention(channels)

    def forward(self, x):
        main_features, sub_features = torch.split(x, [self.main_channels, self.sub_channels], dim=1)
        processed_main = self.main_branch_conv3(self.main_branch_conv2(self.main_branch_conv1(main_features)))
        processed_sub = self.sub_branch_conv(sub_features)
        return self.spatial_attention(processed_sub) * processed_main + self.channel_attention(
            processed_main) * processed_sub


# ===================== SKFF / Wide_Transformer =====================
class SKFF(nn.Module):
    def __init__(self, in_channels, height=3, reduction=8, bias=False):
        super(SKFF, self).__init__()
        self.height = height
        d = max(int(in_channels / reduction), 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, bias=bias), nn.PReLU())
        self.fcs = nn.ModuleList([nn.Conv2d(d, in_channels, 1, bias=bias) for _ in range(self.height)])
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        batch_size = inp_feats[0].shape[0]
        n_feats = inp_feats[0].shape[1]
        inp_feats = torch.cat(inp_feats, dim=1).view(batch_size, self.height, n_feats, inp_feats[0].shape[2],
                                                     inp_feats[0].shape[3])
        feats_U = torch.sum(inp_feats, dim=1)
        attention_vectors = self.softmax(
            torch.cat([fc(self.conv_du(self.avg_pool(feats_U))) for fc in self.fcs], dim=1).view(batch_size,
                                                                                                 self.height, n_feats,
                                                                                                 1, 1))
        return torch.sum(inp_feats * attention_vectors, dim=1)


class WideChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, max(in_channels // reduction, 1), 1)
        self.fc2 = nn.Conv2d(max(in_channels // reduction, 1), in_channels, 1)

    def forward(self, x):
        return x * torch.sigmoid(self.fc2(F.relu(self.fc1(self.global_avg_pool(x)))))


class Wide_Transformer(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 4, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 4, dim * 4, kernel_size=3, padding=1, groups=dim * 4, bias=bias)
        self.project_out_1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.channel = WideChannelAttention(dim)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v, l = self.qkv_dwconv(self.qkv(x)).chunk(4, dim=1)
        q = F.normalize(rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads), dim=-1)
        k = F.normalize(rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads), dim=-1)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        out = rearrange(attn.softmax(dim=-1) @ v, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        return self.project_out_1(out + self.channel(l)) + x


# ===================== SFFNet 基础卷积模块 =====================
class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2, bias=bias),
            norm_layer(out_channels)
        )


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, norm_layer=nn.BatchNorm2d):
        super().__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=kernel_size // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


# ===================== SFFNet 高级模块 =====================
class GlBlock(nn.Module):
    def __init__(self, dim, num_heads=8, drop_path=0.):
        super().__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type='WithBias')
        self.attn = Wide_Transformer(dim, num_heads=num_heads, bias=False)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = LayerNorm(dim, LayerNorm_type='WithBias')
        self.down = nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x):
        x = self.down(x)
        return self.norm2(x + self.drop_path(self.attn(self.norm1(x))))


class multilocalBlock(nn.Module):
    def __init__(self, dim, outdim, window_size=8, drop_path=0.):
        super().__init__()
        self.down = nn.Conv2d(dim, outdim, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(outdim)
        self.attn = SeparableConvBN(outdim, outdim, kernel_size=window_size)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm2d(outdim)

    def forward(self, x):
        x = self.down(x)
        attn_out = self.attn(self.norm1(x))
        if attn_out.shape[2:] != x.shape[2:]:
            attn_out = attn_out[:, :, :x.shape[2], :x.shape[3]]
        return self.drop_path(self.norm2(x + self.drop_path(attn_out)))


class FMS(nn.Module):
    def __init__(self, channels, num_heads=8, window_size=8):
        super().__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.glb = GlBlock(dim=channels, num_heads=num_heads)
        self.localb = multilocalBlock(dim=channels, outdim=channels, window_size=window_size)
        self.skff = SKFF(in_channels=channels, height=3, reduction=8)

        self.out_L = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.BatchNorm2d(channels), nn.ReLU(True))
        self.out_H = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.BatchNorm2d(channels), nn.ReLU(True))
        self.out_glb = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.BatchNorm2d(channels), nn.ReLU(True))
        self.out_local = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.BatchNorm2d(channels), nn.ReLU(True))

    def forward(self, x):
        yL, yH = self.wt(x)
        yH_fused = self.skff([yH[0][:, :, 0], yH[0][:, :, 1], yH[0][:, :, 2]])
        return self.out_L(yL), self.out_H(yH_fused), self.out_glb(self.glb(x)), self.out_local(self.localb(x))


class MDAF(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.norm1 = LayerNorm(dim, 'WithBias')
        self.norm2 = LayerNorm(dim, 'WithBias')
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1)

        # 【恢复精度核心】：加回 21 大卷积核捕捉长距离边界，采用 groups=dim 后参数量增加为 0 !
        self.conv1_ops = nn.ModuleList([
            nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim),
            nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim),
            nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim),  # 恢复
            nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim),
            nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim),
            nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)  # 恢复
        ])
        self.conv2_ops = nn.ModuleList([
            nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim),
            nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim),
            nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim),  # 恢复
            nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim),
            nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim),
            nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)  # 恢复
        ])

    def forward(self, x1, x2):
        b, c, h, w = x1.shape
        out1 = self.project_out(sum(conv(self.norm1(x1)) for conv in self.conv1_ops))
        out2 = self.project_out(sum(conv(self.norm2(x2)) for conv in self.conv2_ops))

        k1 = v1 = rearrange(out1, 'b (head c) h w -> b head h (w c)', head=self.num_heads)
        k2 = v2 = rearrange(out2, 'b (head c) h w -> b head w (h c)', head=self.num_heads)
        q2 = rearrange(out1, 'b (head c) h w -> b head w (h c)', head=self.num_heads)
        q1 = rearrange(out2, 'b (head c) h w -> b head h (w c)', head=self.num_heads)

        out3 = (F.normalize(q1, dim=-1) @ F.normalize(k1, dim=-1).transpose(-2, -1)).softmax(dim=-1) @ v1 + q1
        out4 = (F.normalize(q2, dim=-1) @ F.normalize(k2, dim=-1).transpose(-2, -1)).softmax(dim=-1) @ v2 + q2

        out3 = rearrange(out3, 'b head h (w c) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out4 = rearrange(out4, 'b head w (h c) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        return self.project_out(out3) + self.project_out(out4) + x1 + x2


class CAFM(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.spatial_attn = nn.Sequential(nn.Conv2d(2, 1, 3, padding=1), nn.ReLU(), nn.Conv2d(1, 1, 3, padding=1))
        self.avg_max_conv = nn.Conv2d(channel, channel, 1)
        self.fusion = nn.Conv2d(channel * 2, channel, 1)

    def forward(self, f1, f2):
        a1_sp = F.sigmoid(self.spatial_attn(torch.cat([torch.mean(f1, 1, True), torch.max(f1, 1, True)[0]], 1)))
        a2_sp = F.sigmoid(self.spatial_attn(torch.cat([torch.mean(f2, 1, True), torch.max(f2, 1, True)[0]], 1)))

        f1_out = f1 * a1_sp + f1
        f2_out = f2 * a2_sp + f2
        return self.fusion(torch.cat((f1_out, f2_out), dim=1))


# ===================== SFFNeck (扩容的高精度重构版) =====================
class SFFNeck(nn.Module):
    def __init__(self, in_channels, sff_decode_channels=96, window_size=8, num_features=6):
        super().__init__()
        self.convs = nn.ModuleList([ConvBN(in_channels, sff_decode_channels, 1) for _ in range(num_features)])

        # 【扩容核心】：不压得太死。将 576 通道压到 192 而不是 96。计算量只有原版1/9，但比之前96宽裕两倍。
        self.mid_channels = 192
        self.bottleneck = nn.Sequential(
            nn.Conv2d(num_features * sff_decode_channels, self.mid_channels, 1, bias=False),
            nn.BatchNorm2d(self.mid_channels),
            nn.ReLU(inplace=True)
        )

        self.fcm = FCM(channels=self.mid_channels)
        self.fuseFeature = FMS(channels=self.mid_channels, num_heads=8, window_size=window_size)
        self.MDAF_L = MDAF(self.mid_channels, num_heads=8)
        self.MDAF_H = MDAF(self.mid_channels, num_heads=8)
        self.cafm = CAFM(channel=self.mid_channels)

        # 最后融合后，映射回主干所需要的 96 通道
        self.out_proj = ConvBN(self.mid_channels, sff_decode_channels, kernel_size=1)

    def forward(self, features):
        feats = [self.convs[i](f) for i, f in enumerate(features)]
        x = self.bottleneck(torch.cat(feats, dim=1))

        x = self.fcm(x)
        fusefeature_L, fusefeature_H, glb, local = self.fuseFeature(x)

        Fmh = self.MDAF_H(fusefeature_H, local)
        Flg = self.MDAF_L(fusefeature_L, glb)

        fused = self.cafm(Fmh, Flg)
        return self.out_proj(fused)


# ===================== DPTHead =====================
def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(features, nn.ReLU(False), deconv=False, bn=use_bn, expand=False, align_corners=True,
                              size=size)


class DPTHead(nn.Module):
    def __init__(self, nclass, features=256, use_bn=False):
        super().__init__()
        self.resize_layers = nn.ModuleList([
            nn.Sequential(ConvBN(96, features, 1), nn.ConvTranspose2d(features, features, 4, 4, 0)),
            nn.Sequential(ConvBN(96, features, 1), nn.ConvTranspose2d(features, features, 2, 2, 0)),
            ConvBN(96, features, 1),
            nn.Sequential(ConvBN(96, features, 1), nn.Conv2d(features, features, 3, 2, 1))
        ])

        self.pfesa = PFESA(base_ratio=0.1)
        self.scratch = _make_scratch([features] * 4, features, groups=1, expand=False)
        self.scratch.stem_transpose = None
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, features, 3, 1, 1), nn.ReLU(True), nn.Conv2d(features, nclass, 1)
        )

    def forward(self, x1, x2, x3, x4):
        layer_1_rn = self.pfesa(self.scratch.layer1_rn(self.resize_layers[0](x1)))
        layer_2_rn = self.pfesa(self.scratch.layer2_rn(self.resize_layers[1](x2)))
        layer_3_rn = self.pfesa(self.scratch.layer3_rn(self.resize_layers[2](x3)))
        layer_4_rn = self.pfesa(self.scratch.layer4_rn(self.resize_layers[3](x4)))

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        return self.scratch.output_conv(path_1)


# ===================== DPT 模型顶层 =====================
class DPT(nn.Module):
    def __init__(self, encoder_size='s', nclass=1, features=128, use_bn=False, backbone=None):
        super().__init__()
        self.layer_indices = [1, 2, 3, 4, 5, 7, 11]
        self.backbone = backbone
        embed_dim = getattr(self.backbone, 'embed_dim', 384)

        self.proj_x1 = ConvBN(embed_dim, 96, kernel_size=1)
        self.proj_x2 = ConvBN(embed_dim * 2, 96, kernel_size=1)
        self.proj_x3 = ConvBN(embed_dim * 2, 96, kernel_size=1)
        self.proj_x4 = ConvBN(embed_dim * 2, 96, kernel_size=1)

        self.neck = SFFNeck(in_channels=embed_dim, sff_decode_channels=96, window_size=8, num_features=6)

        self.upsample_smooth = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=3, padding=1, groups=96, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )

        self.head = DPTHead(nclass=nclass, features=features, use_bn=use_bn)

        # ---------------- 初始化结束时打印参数量 ----------------
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("\n" + "=" * 50)
        print("🚀 DPT Model Successfully Initialized!")
        print(f"📊 总参数量 (Total Params):      {total_params / 1e6:.2f} M")
        print(f"🔥 可训练参数量 (Trainable Params): {trainable_params / 1e6:.2f} M")
        print("=" * 50 + "\n")

    def forward(self, x):
        patch_h, patch_w = x.shape[-2] // 16, x.shape[-1] // 16
        out_features = self.backbone.get_intermediate_layers(x, n=self.layer_indices)

        processed = []
        for feat in out_features:
            if feat.ndim == 3 and feat.shape[1] == patch_h * patch_w + 1:
                feat = feat[:, 1:]
            processed.append(feat.permute(0, 2, 1).reshape(feat.shape[0], feat.shape[-1], patch_h, patch_w))

        x1_base = self.proj_x1(processed[0])
        x2_base = self.proj_x2(torch.cat([processed[1], processed[2]], dim=1))
        x3_base = self.proj_x3(torch.cat([processed[3], processed[4]], dim=1))
        x4_base = self.proj_x4(torch.cat([processed[5], processed[6]], dim=1))

        sff_global_feat = self.neck(processed[1:])

        sff_global_feat_up = F.interpolate(sff_global_feat, size=(patch_h, patch_w), mode='bilinear',
                                           align_corners=False)
        sff_global_feat_up = self.upsample_smooth(sff_global_feat_up)

        out = self.head(x1_base, x2_base + sff_global_feat_up, x3_base + sff_global_feat_up,
                        x4_base + sff_global_feat_up)
        return F.interpolate(out, (x.shape[-2], x.shape[-1]), mode='bilinear', align_corners=True)