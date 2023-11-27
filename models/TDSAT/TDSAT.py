from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F


""" Skip Connection """

class AdaptiveSkipConnection(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.conv_weight = nn.Sequential(
            nn.Conv3d(channel * 2, channel, 1),
            nn.Tanh(),
            nn.Conv3d(channel, channel, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        w = self.conv_weight(torch.cat([x, y], dim=1))
        return (1 - w) * x + w * y


""" Spectral Enhancement Module"""


class GlobalAvgPool3d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        sum = torch.sum(x, dim=(2, 3, 4), keepdim=True)
        return sum / (x.shape[2] * x.shape[3] * x.shape[4])


class SpectralEnhancement(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.se = nn.Sequential(
            GlobalAvgPool3d(),
            nn.Conv3d(ch, ch, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.se(x) * x




""" Spatial-Spectral Attention Transformer Block """

class MHSA3D(nn.Module):
    def __init__(self, channels=16, num_heads=1):
        super(MHSA3D, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.qkv = nn.Conv3d(channels, channels * 3, kernel_size=(1, 1, 1), bias=False)
        self.qkv_conv = nn.Conv3d(channels * 3, channels * 3, kernel_size=(1, 3, 3), padding=(0, 1, 1), groups=channels * 3, bias=False)# 331
        self.project_out = nn.Conv3d(channels, channels, kernel_size=(1, 1, 1), bias=False)

    def forward(self, x):
        b, t, c, h, w = x.shape
        q, k, v = self.qkv_conv(self.qkv(x)).chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, -1,  h * w * t)
        k = k.reshape(b, self.num_heads, -1,  h * w * t)
        v = v.reshape(b, self.num_heads, -1,  h * w * t)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)

        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
        out = self.project_out(torch.matmul(attn, v).reshape(b, t, -1, h, w))
        return out


class GDFN3D(nn.Module):
    def __init__(self, channels, expansion_factor):
        super(GDFN3D, self).__init__()

        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Conv3d(channels, hidden_channels * 2, kernel_size=(1, 1, 1), bias=False)

        self.conv = nn.Conv3d(hidden_channels * 2, hidden_channels * 2, kernel_size=(3, 3, 3), padding=(1, 1, 1), groups=hidden_channels * 2, bias=False)# 331


        self.project_out = nn.Conv3d(hidden_channels,  channels, kernel_size=(1, 1, 1), bias=False)

    def forward(self, x):
        x1, x2 = self.conv(self.project_in(x)).chunk(2, dim=1)
        x = self.project_out(F.gelu(x1) * x2)
        return x

class GCFN3D(nn.Module):
    def __init__(self, channels, expansion_factor):
        super(GCFN3D, self).__init__()

        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Conv3d(channels, hidden_channels * 2, kernel_size=(1, 1, 1), bias=False)
        self.conv = nn.Conv3d(hidden_channels * 2, hidden_channels * 2, kernel_size=(1, 3, 3), padding=(0, 1, 1), groups=1, bias=False) #331
        self.project_out = nn.Conv3d(hidden_channels,  channels, kernel_size=(1, 1, 1), bias=False)

    def forward(self, x):
        x1, x2 = self.conv(self.project_in(x)).chunk(2, dim=1)
        x = self.project_out(F.gelu(x1) * x2)
        return x


class SATB(nn.Module):
    def __init__(self, channels, num_heads=1, expansion_factor=2.66):
        super(SATB, self).__init__()
        self.se = SpectralEnhancement(channels)
        self.norm1 = nn.LayerNorm(channels)
        self.attn = MHSA3D(channels, num_heads)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = GDFN3D(channels, expansion_factor)


    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x + self.attn(self.norm1(x.reshape(b, t, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                          .contiguous().reshape(b, t, c, h, w))
        x = x + self.ffn(self.norm2(x.reshape(b, t, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
        .contiguous().reshape(b, t, c, h, w))
        x = self.se(x)
        return x

class EmbeddingBlock(nn.Module):
    def __init__(self, channels, num_heads=1, expansion_factor=2.66):
        super(EmbeddingBlock, self).__init__()
        self.se = SpectralEnhancement(channels)
        self.norm1 = nn.LayerNorm(channels)
        self.attn = MHSA3D(channels, num_heads)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = GDFN3D(channels, expansion_factor)
        self.norm3 = nn.LayerNorm(channels)
        self.gcfn = GCFN3D(channels, expansion_factor)

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x + self.attn(self.norm1(x.reshape(b, t, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                          .contiguous().reshape(b, t, c, h, w))
        x = x + self.ffn(self.norm2(x.reshape(b, t, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
        .contiguous().reshape(b, t, c, h, w))
        x = x + self.gcfn(self.norm3(x.reshape(b, t, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                         .contiguous().reshape(b, t, c, h, w))
        return x


""" Three-Dimension Spatial-Spectral Attention Transformer Network """

class Encoder(nn.Module):
    def __init__(self, channels, num_half_layer, sample_idx, Attn=None):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_half_layer):
            if i not in sample_idx:
                encoder_layer = nn.Sequential(OrderedDict([
                    ('conv', nn.Conv3d(channels, channels, 3, 1, 1, bias=False)),
                    ('attn', Attn(channels)) if Attn else None
                ]))
            else:
                encoder_layer = nn.Sequential(OrderedDict([
                    ('conv', nn.Conv3d(channels, channels * 2, 3, (1, 2, 2), 1, bias=False)),
                    ('attn', Attn(channels * 2)) if Attn else None
                ]))
                channels *= 2
            self.layers.append(encoder_layer)

    def forward(self, x, xs):
        num_half_layer = len(self.layers)
        for i in range(num_half_layer - 1):
            x = self.layers[i](x)
            xs.append(x)
        x = self.layers[-1](x)
        return x


class Decoder(nn.Module):
    def __init__(self, channels, num_half_layer, sample_idx, Fusion=None, Attn=None):
        super(Decoder, self).__init__()

        self.layers = nn.ModuleList()
        self.enable_fusion = Fusion is not None

        if self.enable_fusion:
            self.fusions = nn.ModuleList()
            ch = channels
            for i in reversed(range(num_half_layer)):
                fusion_layer = Fusion(ch)
                if i in sample_idx:
                    ch //= 2
                self.fusions.append(fusion_layer)

        for i in reversed(range(num_half_layer)):
            if i not in sample_idx:
                decoder_layer = nn.Sequential(OrderedDict([
                    ('conv', nn.ConvTranspose3d(channels, channels, 3, 1, 1, bias=False)),
                    ('attn', Attn(channels)) if Attn else None
                ]))
            else:
                decoder_layer = nn.Sequential(OrderedDict([
                    ('up', nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=True)),
                    ('conv', nn.Conv3d(channels, channels // 2, 3, 1, 1, bias=False)),
                    ('attn', Attn(channels // 2)) if Attn else None
                ]))
                channels //= 2
            self.layers.append(decoder_layer)

    def forward(self, x, xs):
        num_half_layer = len(self.layers)
        x = self.layers[0](x)
        for i in range(1, num_half_layer):
            if self.enable_fusion:
                x = self.fusions[i](x, xs.pop())
            else:
                x = x + xs.pop()
            x = self.layers[i](x)
        return x

class TDSAT(nn.Module):
    def __init__(
        self,
        in_channels=1,
        channels=16,
        num_half_layer=5,
        sample_idx=[1, 3],
        Attn=SATB,
        Embedding=EmbeddingBlock,
        Fusion=AdaptiveSkipConnection,
    ):
        super(TDSAT, self).__init__()

        self.head = nn.Sequential(
            nn.Conv3d(in_channels, channels, 3, 1, 1, bias=False),
            Embedding(channels) if Embedding else None
        )

        self.encoder = Encoder(channels, num_half_layer, sample_idx, Attn)
        self.decoder = Decoder(channels * (2**len(sample_idx)), num_half_layer, sample_idx,
                               Fusion=Fusion, Attn=Attn)

        self.tail = nn.Sequential(
            nn.ConvTranspose3d(channels, in_channels, 3, 1, 1, bias=True),
            Embedding(in_channels) if Embedding else None
        )

    def forward(self, x):
        xs = [x]
        out = self.head(xs[0])
        xs.append(out)
        out = self.encoder(out, xs)
        out = self.decoder(out, xs)
        out = out + xs.pop()
        out = self.tail(out)
        out = out + xs.pop()
        return out


if __name__ == "__main__":

    net = TDSAT(1, 16, 5, [1, 3])
    x = torch.randn(1,  1, 31, 64, 64)
    output = net(x)
    print(output.shape)