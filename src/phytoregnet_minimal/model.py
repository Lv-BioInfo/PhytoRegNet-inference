"""Inference-only PhytoRegNet architecture for the released checkpoints."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNormAct1D(nn.Module):
    """One-dimensional convolution followed by normalization, activation and dropout."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilation=1,
        norm_type="bn",
        act="relu",
        dropout=0.0,
    ):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        if norm_type == "bn":
            self.norm = nn.BatchNorm1d(out_channels)
        elif norm_type == "gn":
            num_groups = min(8, out_channels)
            while out_channels % num_groups != 0 and num_groups > 1:
                num_groups -= 1
            self.norm = nn.GroupNorm(num_groups, out_channels)
        else:
            self.norm = nn.Identity()
        self.act = (
            nn.ReLU(inplace=True)
            if act == "relu"
            else (
                nn.GELU()
                if act == "gelu"
                else nn.SiLU(inplace=True) if act == "silu" else nn.Identity()
            )
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.dropout(self.act(self.norm(self.conv(x))))


class SEBlock1D(nn.Module):
    """Reweight channels using pooled features and a sigmoid gate."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Conv1d(channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv1d(hidden, channels, kernel_size=1)

    def forward(self, x):
        w = self.pool(x)
        w = F.relu(self.fc1(w), inplace=True)
        w = torch.sigmoid(self.fc2(w))
        return x * w


class ResidualBlock1D(nn.Module):
    """Residual convolutional refinement with optional channel attention."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=7,
        dilation=1,
        norm_type="bn",
        act="relu",
        dropout=0.0,
        use_se=False,
        use_depthwise=False,
    ):
        super().__init__()
        if use_depthwise and in_channels == out_channels:
            self.conv1 = DepthwiseDilatedConv1D(
                in_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                norm_type=norm_type,
                act=act,
                dropout=dropout,
            )
            self.conv2 = DepthwiseDilatedConv1D(
                out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                norm_type=norm_type,
                act="none",
                dropout=0.0,
            )
        else:
            self.conv1 = ConvNormAct1D(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                norm_type=norm_type,
                act=act,
                dropout=dropout,
            )
            self.conv2 = ConvNormAct1D(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                norm_type=norm_type,
                act="none",
                dropout=0.0,
            )
        self.se = SEBlock1D(out_channels) if use_se else nn.Identity()
        self.skip = (
            nn.Conv1d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.out_act = (
            nn.ReLU(inplace=True)
            if act == "relu"
            else nn.GELU() if act == "gelu" else nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return self.out_act(self.se(self.conv2(self.conv1(x))) + self.skip(x))


class DownBlock1D(nn.Module):
    """Refine encoder features, return a skip connection and downsample by two."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=7,
        norm_type="bn",
        act="relu",
        dropout=0.0,
        use_se=False,
    ):
        super().__init__()

        def RBlock(in_c, out_c, k=kernel_size):
            return ResidualBlock1D(
                in_c,
                out_c,
                kernel_size=k,
                norm_type=norm_type,
                act=act,
                dropout=dropout,
                use_se=use_se,
            )

        self.refine = nn.Sequential(
            RBlock(in_channels, out_channels), RBlock(out_channels, out_channels)
        )
        self.down = nn.Sequential(
            DepthwiseDilatedConv1D(
                out_channels,
                kernel_size=3,
                dilation=1,
                norm_type=norm_type,
                act=act,
                dropout=0.0,
            ),
            nn.MaxPool1d(kernel_size=2),
        )

    def forward(self, x):
        skip = self.refine(x)
        down = self.down(skip)
        return (skip, down)


class UpBlock1D(nn.Module):
    """Interpolate decoder features and refine their gated skip concatenation."""

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        kernel_size=7,
        norm_type="bn",
        act="relu",
        dropout=0.0,
        use_se=False,
        use_gated_skip=True,
    ):
        super().__init__()
        self.use_gated_skip = use_gated_skip
        self.skip_gate = (
            GatedSkipFusion1D(in_channels, skip_channels) if use_gated_skip else None
        )

        def RBlock(in_c, out_c, k=kernel_size):
            return ResidualBlock1D(
                in_c,
                out_c,
                kernel_size=k,
                norm_type=norm_type,
                act=act,
                dropout=dropout,
                use_se=use_se,
            )

        self.fuse = nn.Sequential(
            RBlock(in_channels + skip_channels, out_channels),
            RBlock(out_channels, out_channels),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.size(-1), mode="linear", align_corners=False)
        if self.use_gated_skip:
            skip = self.skip_gate(x, skip)
        x = torch.cat([x, skip], dim=1)
        return self.fuse(x)


class GatedSkipFusion1D(nn.Module):
    """Gate encoder features using both encoder and decoder representations."""

    def __init__(self, up_channels: int, skip_channels: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv1d(up_channels + skip_channels, skip_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, up_feat, skip_feat):
        gate = self.gate(torch.cat([up_feat, skip_feat], dim=1))
        skip_feat = skip_feat * gate
        return skip_feat


class RotaryPositionalEmbedding(nn.Module):
    """Apply rotary positional encoding to attention queries and keys."""

    def __init__(self, head_dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even, got {head_dim}")
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base
        inv_freq = 1.0 / base ** (torch.arange(0, head_dim, 2).float() / head_dim)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=torch.float32, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cache", emb.cos(), persistent=False)
        self.register_buffer("sin_cache", emb.sin(), persistent=False)

    @staticmethod
    def _rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q, k):
        L = q.shape[-2]
        if L > self.cos_cache.shape[0]:
            self._build_cache(L)
            self.cos_cache = self.cos_cache.to(q.device)
            self.sin_cache = self.sin_cache.to(q.device)
        cos = self.cos_cache[:L].to(q.dtype)
        sin = self.sin_cache[:L].to(q.dtype)
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return (q_rot, k_rot)


class MultiheadAttentionRoPE(nn.Module):
    """Compute multi-head self-attention with rotary query and key encoding."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len=1024)
        self.attn_dropout = dropout

    def forward(self, x):
        B, L, D = x.shape
        H = self.n_heads
        Dh = self.head_dim
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, L, H, Dh).transpose(1, 2)
        k = k.view(B, L, H, Dh).transpose(1, 2)
        v = v.view(B, L, H, Dh).transpose(1, 2)
        q, k = self.rope(q, k)
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(out)
        return out


class TransformerBlock(nn.Module):
    """Pre-normalized self-attention and feed-forward residual block."""

    def __init__(
        self,
        d_model: int = 192,
        n_heads: int = 8,
        ffn_dim: int = 768,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        drop_path: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiheadAttentionRoPE(d_model, n_heads, dropout=attn_dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x


class DropPath(nn.Module):
    """Apply stochastic depth; return input unchanged in evaluation mode."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x * mask / keep_prob


class DepthwiseDilatedConv1D(nn.Module):
    """Depthwise and pointwise convolutions with separate normalization and activation."""

    def __init__(
        self,
        channels,
        kernel_size=3,
        dilation=1,
        norm_type="bn",
        act="relu",
        dropout=0.1,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.dw = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(channels) if norm_type == "bn" else nn.Identity()
        self.pw = nn.Conv1d(channels, channels, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(channels) if norm_type == "bn" else nn.Identity()
        self.act_fn = nn.GELU() if act == "gelu" else nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.act_fn(self.bn1(self.dw(x)))
        x = self.act_fn(self.bn2(self.pw(x)))
        return self.drop(x)


class MultiScaleBottleneck1D(nn.Module):
    """Fuse dilated convolution branches and a pooled global branch."""

    def __init__(
        self,
        in_channels,
        out_channels,
        branch_channels=128,
        dilations=(1, 2, 4, 8),
        norm_type="bn",
        act="relu",
        dropout=0.0,
        use_se=True,
    ):
        super().__init__()
        self.proj = ConvNormAct1D(
            in_channels,
            out_channels,
            kernel_size=1,
            norm_type=norm_type,
            act=act,
            dropout=0.0,
        )
        self.branches = nn.ModuleList(
            [
                ConvNormAct1D(
                    out_channels,
                    branch_channels,
                    kernel_size=3,
                    dilation=d,
                    norm_type=norm_type,
                    act=act,
                    dropout=dropout,
                )
                for d in dilations
            ]
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.global_proj = ConvNormAct1D(
            out_channels,
            branch_channels,
            kernel_size=1,
            norm_type="none",
            act=act,
            dropout=0.0,
        )
        total_channels = branch_channels * (len(dilations) + 1)
        self.fuse = nn.Sequential(
            ConvNormAct1D(
                total_channels,
                out_channels,
                kernel_size=1,
                norm_type=norm_type,
                act=act,
                dropout=0.0,
            ),
            ResidualBlock1D(
                out_channels,
                out_channels,
                kernel_size=5,
                norm_type=norm_type,
                act=act,
                dropout=dropout,
                use_se=use_se,
            ),
        )
        self.use_residual = True
        self.out_channels = out_channels

    def forward(self, x):
        x_proj = self.proj(x)
        feats = [b(x_proj) for b in self.branches]
        g = F.interpolate(
            self.global_proj(self.global_pool(x_proj)),
            size=x_proj.size(-1),
            mode="linear",
            align_corners=False,
        )
        feats.append(g)
        x_fuse = self.fuse(torch.cat(feats, dim=1))
        return x_proj + x_fuse if self.use_residual else x_fuse


class SeqUNet1D(nn.Module):
    """Predict tissue accessibility profiles and regional signal from DNA.

    Input is a floating-point tensor shaped (batch, 4, length) or
    (batch, length, 4), with nucleotide channels in A, C, G, T order.
    Returns profiles shaped (batch, tissues, bins) and log1p regional
    signal shaped (batch, tissues). Call eval() before inference.
    """

    def __init__(self, model_params):
        super().__init__()
        self.inputlen = model_params.get("inputlen", 10000)
        self.target_region_len = model_params.get("target_region_len", 4000)
        self.bin_size = model_params.get("bin_size", 5)
        self.n_bins = self.target_region_len // self.bin_size
        self.num_targets = model_params.get("num_targets", 1)
        if self.target_region_len % self.bin_size != 0:
            raise ValueError("target_region_len must be divisible by bin_size")
        if self.n_bins != self.target_region_len // self.bin_size:
            raise ValueError("n_bins must equal target_region_len // bin_size")
        self.base_filters = int(model_params.get("base_filters", 128))
        self.dropout = float(model_params.get("dropout", 0.1))
        self.norm_type = model_params.get("norm_type", "bn")
        self.act = model_params.get("act", "relu")
        self.use_se = bool(model_params.get("use_se", True))
        self.use_gated_skip = bool(model_params.get("use_gated_skip", True))
        self.branch_channels = int(model_params.get("branch_channels", 128))
        self.use_transformer = bool(model_params.get("use_transformer", True))
        self.tf_d_model = int(model_params.get("tf_d_model", 192))
        self.tf_n_heads = int(model_params.get("tf_n_heads", 8))
        self.tf_n_layers = int(model_params.get("tf_n_layers", 4))
        self.tf_ffn_dim = int(model_params.get("tf_ffn_dim", self.tf_d_model * 4))
        self.tf_dropout = float(model_params.get("tf_dropout", 0.1))
        self.tf_attn_dropout = float(model_params.get("tf_attn_dropout", 0.1))
        self.tf_drop_path = float(model_params.get("tf_drop_path", 0.1))
        C = self.base_filters
        self.stem = ConvNormAct1D(
            4, C, kernel_size=21, norm_type=self.norm_type, act=self.act
        )
        self.enc1 = DownBlock1D(
            C,
            C,
            kernel_size=7,
            norm_type=self.norm_type,
            act=self.act,
            dropout=self.dropout,
            use_se=self.use_se,
        )
        self.enc2 = DownBlock1D(
            C,
            2 * C,
            kernel_size=7,
            norm_type=self.norm_type,
            act=self.act,
            dropout=self.dropout,
            use_se=self.use_se,
        )
        self.enc3 = DownBlock1D(
            2 * C,
            4 * C,
            kernel_size=7,
            norm_type=self.norm_type,
            act=self.act,
            dropout=self.dropout,
            use_se=self.use_se,
        )
        self.bottleneck = MultiScaleBottleneck1D(
            4 * C,
            8 * C,
            branch_channels=self.branch_channels,
            dilations=(1, 2, 4, 8),
            norm_type=self.norm_type,
            act=self.act,
            dropout=self.dropout,
            use_se=self.use_se,
        )
        self.up3 = UpBlock1D(
            8 * C,
            4 * C,
            4 * C,
            kernel_size=7,
            norm_type=self.norm_type,
            act=self.act,
            dropout=self.dropout,
            use_se=self.use_se,
            use_gated_skip=self.use_gated_skip,
        )
        self.up2 = UpBlock1D(
            4 * C,
            2 * C,
            2 * C,
            kernel_size=7,
            norm_type=self.norm_type,
            act=self.act,
            dropout=self.dropout,
            use_se=self.use_se,
            use_gated_skip=self.use_gated_skip,
        )
        self.up1 = UpBlock1D(
            2 * C,
            C,
            C,
            kernel_size=7,
            norm_type=self.norm_type,
            act=self.act,
            dropout=self.dropout,
            use_se=self.use_se,
            use_gated_skip=self.use_gated_skip,
        )
        self.refine = nn.Sequential(
            ResidualBlock1D(
                C,
                C,
                kernel_size=7,
                norm_type=self.norm_type,
                act=self.act,
                dropout=self.dropout,
                use_se=self.use_se,
            ),
            ResidualBlock1D(
                C,
                C,
                kernel_size=7,
                norm_type=self.norm_type,
                act=self.act,
                dropout=self.dropout,
                use_se=self.use_se,
            ),
        )
        self.target_refine = nn.Sequential(
            ResidualBlock1D(
                C,
                C,
                kernel_size=9,
                norm_type=self.norm_type,
                act=self.act,
                dropout=self.dropout,
                use_se=self.use_se,
                use_depthwise=True,
            ),
            ResidualBlock1D(
                C,
                C,
                kernel_size=7,
                norm_type=self.norm_type,
                act=self.act,
                dropout=self.dropout,
                use_se=self.use_se,
                use_depthwise=True,
            ),
        )
        self.compress_4000_to_2000 = nn.Sequential(
            ResidualBlock1D(
                C,
                C,
                kernel_size=7,
                norm_type=self.norm_type,
                act=self.act,
                dropout=self.dropout,
                use_se=self.use_se,
            ),
            nn.Conv1d(C, C, kernel_size=4, stride=2, padding=1),
        )
        self.compress_2000_to_1000 = nn.Sequential(
            ResidualBlock1D(
                C,
                C,
                kernel_size=5,
                norm_type=self.norm_type,
                act=self.act,
                dropout=self.dropout,
                use_se=self.use_se,
            ),
            nn.Conv1d(C, C, kernel_size=6, stride=2, padding=2),
        )
        self.compress_1000_to_800 = nn.Sequential(
            ResidualBlock1D(
                C,
                C,
                kernel_size=5,
                norm_type=self.norm_type,
                act=self.act,
                dropout=self.dropout,
                use_se=self.use_se,
            ),
            nn.AdaptiveAvgPool1d(self.n_bins),
        )
        self.bin_refine = nn.Sequential(
            ResidualBlock1D(
                C,
                C,
                kernel_size=5,
                norm_type=self.norm_type,
                act=self.act,
                dropout=self.dropout,
                use_se=self.use_se,
            ),
            ResidualBlock1D(
                C,
                C,
                kernel_size=3,
                norm_type=self.norm_type,
                act=self.act,
                dropout=self.dropout,
                use_se=self.use_se,
            ),
        )
        # Shared modules retain their checkpoint registration aliases.
        self.head_processor = nn.Sequential(
            self.target_refine,
            self.compress_4000_to_2000,
            self.compress_2000_to_1000,
            self.compress_1000_to_800,
            self.bin_refine,
        )
        if self.use_transformer:
            self.tf_in_proj = nn.Conv1d(C, self.tf_d_model, kernel_size=1, bias=False)
            self.tf_in_norm = nn.LayerNorm(self.tf_d_model)
            drop_path_rates = [
                self.tf_drop_path * i / max(self.tf_n_layers - 1, 1)
                for i in range(self.tf_n_layers)
            ]
            self.tf_blocks = nn.ModuleList(
                [
                    TransformerBlock(
                        d_model=self.tf_d_model,
                        n_heads=self.tf_n_heads,
                        ffn_dim=self.tf_ffn_dim,
                        dropout=self.tf_dropout,
                        attn_dropout=self.tf_attn_dropout,
                        drop_path=drop_path_rates[i],
                    )
                    for i in range(self.tf_n_layers)
                ]
            )
            self.tf_out_proj = nn.Conv1d(self.tf_d_model, C, kernel_size=1, bias=False)
            self.tf_residual_scale = nn.Parameter(torch.zeros(1))
        self.profile_head = nn.Sequential(
            ResidualBlock1D(
                C,
                C,
                kernel_size=5,
                norm_type=self.norm_type,
                act=self.act,
                dropout=self.dropout,
                use_se=self.use_se,
            ),
            nn.Conv1d(C, self.num_targets, kernel_size=1, bias=True),
            nn.Softplus(),
        )
        self.counts_head = nn.Sequential(
            ResidualBlock1D(
                C,
                C,
                kernel_size=5,
                norm_type=self.norm_type,
                act=self.act,
                dropout=self.dropout,
                use_se=self.use_se,
            ),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(C, self.num_targets),
        )

    def _apply_transformer(self, x_bins):
        """Refine bin features and add their learned scaled residual."""
        x_tf = self.tf_in_proj(x_bins)
        x_tf = x_tf.permute(0, 2, 1).contiguous()
        x_tf = self.tf_in_norm(x_tf)
        for block in self.tf_blocks:
            x_tf = block(x_tf)
        x_tf = x_tf.permute(0, 2, 1).contiguous()
        x_tf = self.tf_out_proj(x_tf)
        return x_bins + self.tf_residual_scale * x_tf

    def forward_features(self, x: torch.Tensor):
        """Return high-resolution convolutional encoder-decoder features."""
        if x.shape[1] == 4:
            x = x
        elif x.shape[2] == 4:
            x = x.permute(0, 2, 1).contiguous()
        else:
            raise ValueError(f"Cannot infer channel dimension from {x.shape}")
        x = self.stem(x)
        skip1, x = self.enc1(x)
        skip2, x = self.enc2(x)
        skip3, x = self.enc3(x)
        x = self.bottleneck(x)
        x = self.up3(x, skip3)
        x = self.up2(x, skip2)
        x = self.up1(x, skip1)
        feat = self.refine(x)
        return feat

    def forward(self, x: torch.Tensor):
        """Return nonnegative profiles and log1p regional-signal predictions."""
        if x.shape[1] != 4:
            x = x.transpose(1, 2)
        feat = self.forward_features(x)
        L = feat.shape[-1]
        start = (L - self.target_region_len) // 2
        x = feat[:, :, start : start + self.target_region_len]
        x_bins = self.head_processor(x)
        if self.use_transformer:
            x_bins = self._apply_transformer(x_bins)
        profile = self.profile_head(x_bins)
        counts = self.counts_head(x_bins)
        return (profile, counts)
