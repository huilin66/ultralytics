"""YOLOv13-specific building blocks.

The modules in this file are kept separate from the custom multi-attribute
detection and segmentation heads so that adding YOLOv13 does not alter their
behavior.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import C2f, C3
from .conv import Conv, DSConv

__all__ = (
    "DSBottleneck",
    "DSC3k",
    "DSC3k2",
    "AdaHyperedgeGen",
    "AdaHGConv",
    "AdaHGComputation",
    "C3AH",
    "FuseModule",
    "HyperACE",
    "DownsampleConv",
    "FullPAD_Tunnel",
)


class DSBottleneck(nn.Module):
    """Depthwise-separable bottleneck used by YOLOv13."""

    def __init__(self, c1, c2, shortcut=True, e=0.5, k1=3, k2=5, d2=1):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = DSConv(c1, c_, k1, s=1, d=1)
        self.cv2 = DSConv(c_, c2, k2, s=1, d=d2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class DSC3k(C3):
    """C3 block whose bottlenecks use depthwise-separable convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k1=3, k2=5, d2=1):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(
            *(DSBottleneck(c_, c_, shortcut=shortcut, e=1.0, k1=k1, k2=k2, d2=d2) for _ in range(n))
        )


class DSC3k2(C2f):
    """C2f block using YOLOv13 depthwise-separable bottlenecks."""

    def __init__(self, c1, c2, n=1, dsc3k=False, e=0.5, g=1, shortcut=True, k1=3, k2=7, d2=1):
        super().__init__(c1, c2, n, shortcut, g, e)
        if dsc3k:
            self.m = nn.ModuleList(
                DSC3k(
                    self.c,
                    self.c,
                    n=2,
                    shortcut=shortcut,
                    g=g,
                    e=1.0,
                    k1=k1,
                    k2=k2,
                    d2=d2,
                )
                for _ in range(n)
            )
        else:
            self.m = nn.ModuleList(
                DSBottleneck(self.c, self.c, shortcut=shortcut, e=1.0, k1=k1, k2=k2, d2=d2) for _ in range(n)
            )


class AdaHyperedgeGen(nn.Module):
    """Generate adaptive hyperedge participation scores."""

    def __init__(self, node_dim, num_hyperedges, num_heads=4, dropout=0.1, context="both"):
        super().__init__()
        self.num_heads = num_heads
        self.num_hyperedges = num_hyperedges
        self.head_dim = node_dim // num_heads
        self.context = context

        self.prototype_base = nn.Parameter(torch.Tensor(num_hyperedges, node_dim))
        nn.init.xavier_uniform_(self.prototype_base)
        if context in ("mean", "max"):
            self.context_net = nn.Linear(node_dim, num_hyperedges * node_dim)
        elif context == "both":
            self.context_net = nn.Linear(2 * node_dim, num_hyperedges * node_dim)
        else:
            raise ValueError(f"Unsupported context '{context}'. Expected one of: 'mean', 'max', 'both'.")

        self.pre_head_proj = nn.Linear(node_dim, node_dim)
        self.dropout = nn.Dropout(dropout)
        self.scaling = math.sqrt(self.head_dim)

    def forward(self, x):
        b, n, d = x.shape
        if self.context == "mean":
            context = x.mean(dim=1)
        elif self.context == "max":
            context, _ = x.max(dim=1)
        else:
            context = torch.cat([x.mean(dim=1), x.max(dim=1).values], dim=-1)

        prototype_offsets = self.context_net(context).view(b, self.num_hyperedges, d)
        prototypes = self.prototype_base.unsqueeze(0) + prototype_offsets

        x_proj = self.pre_head_proj(x)
        x_heads = x_proj.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        proto_heads = prototypes.view(b, self.num_hyperedges, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        x_heads = x_heads.reshape(b * self.num_heads, n, self.head_dim)
        proto_heads = proto_heads.reshape(b * self.num_heads, self.num_hyperedges, self.head_dim).transpose(1, 2)
        logits = torch.bmm(x_heads, proto_heads) / self.scaling
        logits = logits.view(b, self.num_heads, n, self.num_hyperedges).mean(dim=1)
        return F.softmax(self.dropout(logits), dim=1)


class AdaHGConv(nn.Module):
    """Adaptive hypergraph message passing."""

    def __init__(self, embed_dim, num_hyperedges=16, num_heads=4, dropout=0.1, context="both"):
        super().__init__()
        self.edge_generator = AdaHyperedgeGen(embed_dim, num_hyperedges, num_heads, dropout, context)
        self.edge_proj = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU())
        self.node_proj = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU())

    def forward(self, x):
        a = self.edge_generator(x)
        hyperedges = self.edge_proj(torch.bmm(a.transpose(1, 2), x))
        return self.node_proj(torch.bmm(a, hyperedges)) + x


class AdaHGComputation(nn.Module):
    """Apply adaptive hypergraph reasoning to a 4-D feature map."""

    def __init__(self, embed_dim, num_hyperedges=16, num_heads=8, dropout=0.1, context="both"):
        super().__init__()
        self.embed_dim = embed_dim
        self.hgnn = AdaHGConv(embed_dim, num_hyperedges, num_heads, dropout, context)

    def forward(self, x):
        b, c, h, w = x.shape
        tokens = self.hgnn(x.flatten(2).transpose(1, 2))
        return tokens.transpose(1, 2).view(b, c, h, w)


class C3AH(nn.Module):
    """CSP-style adaptive hypergraph block."""

    def __init__(self, c1, c2, e=1.0, num_hyperedges=8, context="both"):
        super().__init__()
        c_ = int(c2 * e)
        if c_ % 16:
            raise ValueError(f"C3AH hidden channels must be divisible by 16, got {c_}.")
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = AdaHGComputation(c_, num_hyperedges, c_ // 16, 0.1, context)
        self.cv3 = Conv(2 * c_, c2, 1)

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class FuseModule(nn.Module):
    """Fuse three feature levels at the middle resolution."""

    def __init__(self, c_in, channel_adjust):
        super().__init__()
        self.downsample = nn.AvgPool2d(kernel_size=2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv_out = Conv((4 if channel_adjust else 3) * c_in, c_in, 1)

    def forward(self, x):
        x_cat = torch.cat([self.downsample(x[0]), x[1], self.upsample(x[2])], dim=1)
        return self.conv_out(x_cat)


class HyperACE(nn.Module):
    """YOLOv13 Hypergraph-based Adaptive Correlation Enhancement block."""

    def __init__(
        self,
        c1,
        c2,
        n=1,
        num_hyperedges=8,
        dsc3k=True,
        shortcut=False,
        e1=0.5,
        e2=1,
        context="both",
        channel_adjust=True,
    ):
        super().__init__()
        self.c = int(c2 * e1)
        self.cv1 = Conv(c1, 3 * self.c, 1, 1)
        self.cv2 = Conv((4 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            DSC3k(self.c, self.c, 2, shortcut, k1=3, k2=7)
            if dsc3k
            else DSBottleneck(self.c, self.c, shortcut=shortcut)
            for _ in range(n)
        )
        self.fuse = FuseModule(c1, channel_adjust)
        self.branch1 = C3AH(self.c, self.c, e2, num_hyperedges, context)
        self.branch2 = C3AH(self.c, self.c, e2, num_hyperedges, context)

    def forward(self, x):
        y = list(self.cv1(self.fuse(x)).chunk(3, 1))
        out1 = self.branch1(y[1])
        out2 = self.branch2(y[1])
        y.extend(m(y[-1]) for m in self.m)
        y[1] = out1
        y.append(out2)
        return self.cv2(torch.cat(y, 1))


class DownsampleConv(nn.Module):
    """Average-pooling downsample used by the YOLOv13 FullPAD path."""

    def __init__(self, in_channels, channel_adjust=True):
        super().__init__()
        self.downsample = nn.AvgPool2d(kernel_size=2)
        self.channel_adjust = Conv(in_channels, in_channels * 2, 1) if channel_adjust else nn.Identity()

    def forward(self, x):
        return self.channel_adjust(self.downsample(x))


class FullPAD_Tunnel(nn.Module):
    """Learnable gated residual tunnel used by YOLOv13."""

    def __init__(self):
        super().__init__()
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return x[0] + self.gate * x[1]
