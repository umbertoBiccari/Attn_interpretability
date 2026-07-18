import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureTokenizer(nn.Module):
    def __init__(self, in_dim: int, emb_dim: int):
        super().__init__()
        self.value_weight = nn.Parameter(torch.randn(in_dim, emb_dim) * 0.02)
        self.feature_embedding = nn.Parameter(torch.randn(in_dim, emb_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, d]
        x_exp = x.unsqueeze(-1)                           # [B, d, 1]
        tokens = x_exp * self.value_weight.unsqueeze(0)  # [B, d, emb_dim]
        tokens = tokens + self.feature_embedding.unsqueeze(0)
        return tokens


class TransformerEncoderLayerWithAttn(nn.Module):
    """
    Pre-norm transformer encoder block with explicit attention return.
    When return_attn=True, also returns the projected values V.
    """
    def __init__(self, emb_dim: int, nhead: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()

        if emb_dim % nhead != 0:
            raise ValueError(f"emb_dim={emb_dim} must be divisible by nhead={nhead}")

        self.emb_dim = emb_dim
        self.nhead = nhead
        self.head_dim = emb_dim // nhead

        self.self_attn = nn.MultiheadAttention(
            embed_dim=emb_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)

        self.linear1 = nn.Linear(emb_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, emb_dim)

        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_ff = nn.Dropout(dropout)
        self.dropout_act = nn.Dropout(dropout)

    def _extract_values(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract projected values V from the internal MultiheadAttention.
        Input:
            x: [B, N, E]
        Output:
            v: [B, H, N, D]
        """
        B, N, E = x.shape

        # in_proj_weight has shape [3E, E]
        # in_proj_bias has shape [3E]
        qkv = F.linear(
            x,
            self.self_attn.in_proj_weight,
            self.self_attn.in_proj_bias
        )  # [B, N, 3E]

        _, _, v = qkv.chunk(3, dim=-1)  # [B, N, E]

        # reshape to [B, H, N, D]
        v = v.view(B, N, self.nhead, self.head_dim).transpose(1, 2).contiguous()
        return v

    def forward(self, src: torch.Tensor, return_attn: bool = False):
        # Pre-norm attention block
        x = self.norm1(src)  # [B, N, E]

        values = None
        if return_attn:
            values = self._extract_values(x)  # [B, H, N, D]

        attn_out, attn_weights = self.self_attn(
            x, x, x,
            need_weights=return_attn,
            average_attn_weights=False
        )
        src = src + self.dropout_attn(attn_out)

        # Pre-norm FFN block
        x = self.norm2(src)
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout_act(x)
        x = self.linear2(x)
        src = src + self.dropout_ff(x)

        if return_attn:
            # attn_weights: [B, H, N, N]
            # values:       [B, H, N, D]
            return src, attn_weights, values

        return src

class TransformerTrunkWithAttn(nn.Module):
    def __init__(
        self,
        in_dim: int,
        emb_dim: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        ff_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.tokenizer = FeatureTokenizer(in_dim, emb_dim)

        self.layers = nn.ModuleList([
            TransformerEncoderLayerWithAttn(
                emb_dim=emb_dim,
                nhead=nhead,
                ff_dim=ff_dim,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(emb_dim)

    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
        return_hidden: bool = False,
    ):
        z = self.tokenizer(x)  # [B, d, emb_dim]

        attn_layers = []
        value_layers = []
        hidden_layers = []

        if return_hidden:
            hidden_layers.append(z)

        for layer in self.layers:
            if return_attn:
                z, attn, values = layer(z, return_attn=True)
                attn_layers.append(attn)
                value_layers.append(values)
            else:
                z = layer(z, return_attn=False)

            if return_hidden:
                hidden_layers.append(z)

        z = self.norm(z)

        if return_hidden:
            hidden_layers.append(z)  # final normalized representation

        h = z.mean(dim=1)

        if return_attn and return_hidden:
            return h, attn_layers, value_layers, hidden_layers

        if return_attn:
            return h, attn_layers, value_layers

        if return_hidden:
            return h, hidden_layers

        return h

class SingleOutTransformerNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        emb_dim: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        ff_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.trunk = TransformerTrunkWithAttn(
            in_dim=in_dim,
            emb_dim=emb_dim,
            nhead=nhead,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
        )
        self.head = nn.Linear(emb_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
        return_hidden: bool = False,
    ):
        if return_attn and return_hidden:
            h, attn_layers, value_layers, hidden_layers = self.trunk(
                x, return_attn=True, return_hidden=True
            )
            y = self.head(h)
            return y, attn_layers, value_layers, hidden_layers

        if return_attn:
            h, attn_layers, value_layers = self.trunk(x, return_attn=True)
            y = self.head(h)
            return y, attn_layers, value_layers

        if return_hidden:
            h, hidden_layers = self.trunk(x, return_hidden=True)
            y = self.head(h)
            return y, hidden_layers

        h = self.trunk(x)
        return self.head(h)