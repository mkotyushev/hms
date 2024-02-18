import torch
import torch.nn as nn
from einops import repeat


class Mlp(nn.Module):
    def __init__(self, inp_dim, mid_dim, out_dim, dropout):
        super().__init__()
        self.linear_1 = nn.Linear(inp_dim, mid_dim)
        self.linear_2 = nn.Linear(mid_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x


class HmsEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()

        # S
        self.self_attn_s = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout,
            batch_first=True,
        )
        self.cross_attn_s = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout,
            batch_first=True,
        )
        self.mlp_s = Mlp(embed_dim, embed_dim, embed_dim, dropout)

        self.norm_s_1 = nn.LayerNorm(embed_dim)
        self.norm_s_2 = nn.LayerNorm(embed_dim)
        self.norm_s_3 = nn.LayerNorm(embed_dim)

        # E
        self.self_attn_e = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout,
            batch_first=True,
        )
        self.cross_attn_e = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout,
            batch_first=True,
        )
        self.mlp_e = Mlp(embed_dim, embed_dim, embed_dim, dropout)

        self.norm_e_1 = nn.LayerNorm(embed_dim)
        self.norm_e_2 = nn.LayerNorm(embed_dim)
        self.norm_e_3 = nn.LayerNorm(embed_dim)

        # Common
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        x_s, x_e = x
        
        # Self
        x_s = self.norm_s_1(x_s + self.dropout(self.self_attn_s(x_s, x_s, x_s, need_weights=False)[0]))
        x_e = self.norm_e_1(x_e + self.dropout(self.self_attn_e(x_e, x_e, x_e, need_weights=False)[0]))

        # Cross
        # TODO: implement cheap crossvit trick to avoid costly cross attention
        x_s_after_cross = self.norm_s_2(x_s + self.dropout(self.cross_attn_s(x_s, x_e, x_e, need_weights=False)[0]))
        x_e_after_cross = self.norm_e_2(x_e + self.dropout(self.cross_attn_e(x_e, x_s, x_s, need_weights=False)[0]))
        x_s, x_e = x_s_after_cross, x_e_after_cross

        # Mlp
        x_s = self.norm_s_3(x_s + self.mlp_s(x_s))
        x_e = self.norm_e_3(x_e + self.mlp_e(x_e))

        x = x_s, x_e

        return x


def embed(x, patch_embed, pos_embed, cls_token, unk_token):
    B, N, _ = x.shape

    # Patch embed
    x = patch_embed(x)

    # Set patches with nan to unk token
    nan_mask = x.isnan().any(-1)
    n_nans = nan_mask.sum()
    nan_mask = nan_mask[..., None].expand(x.size())
    unk_token = unk_token.expand(n_nans, -1).reshape(-1)
    # TODO: check why explicit bfloat16 conversion is required 
    # for bfloat16 training
    if x.dtype == torch.bfloat16:
        unk_token = unk_token.bfloat16()
    x[nan_mask] = unk_token

    # Cat cls token
    cls_token = repeat(cls_token, '() n d -> b n d', b = B)
    x = torch.cat((cls_token, x), dim=1)

    # Pos embed
    x += pos_embed[:, :(N + 1)]

    return x


class HmsClassifier(nn.Module):
    def __init__(
        self, 
        n_classes,
        input_dim_s, 
        num_patches_s,
        input_dim_e, 
        num_patches_e,
        embed_dim, 
        num_heads, 
        dropout,
        depth,
    ):
        super().__init__()

        # Patch embedding
        self.patch_embed_s = nn.Linear(input_dim_s, embed_dim)
        self.patch_embed_e = nn.Linear(input_dim_e, embed_dim)

        # Pos embedding
        self.pos_embed_s = nn.Parameter(torch.randn(1, num_patches_s + 1, embed_dim))
        self.cls_token_s = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.unk_token_s = nn.Parameter(torch.randn(embed_dim))

        self.pos_embed_e = nn.Parameter(torch.randn(1, num_patches_e + 1, embed_dim))
        self.cls_token_e = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.unk_token_e = nn.Parameter(torch.randn(embed_dim))

        # Common
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = nn.Sequential(
            *[
                HmsEncoderLayer(embed_dim, num_heads, dropout)
                for _ in range(depth)
            ]
        )

        # Mlps
        self.mlp_s = Mlp(embed_dim, embed_dim, n_classes, dropout)
        self.mlp_e = Mlp(embed_dim, embed_dim, n_classes, dropout)
    
    def forward(self, x_s, x_e):
        # Flatten patches

        # s: (B, K=4, T=300, E=100) -> (B, N=1200, E=100)
        B, _, _, E = x_s.shape
        x_s = x_s.reshape(B, -1, E)

        # e: (B, T=50, E=200, F=20) -> (B, N=1000, E=200)
        B, _, E, _ = x_e.shape
        x_e = x_e.permute([0, 1, 3, 2])
        x_e = x_e.reshape(B, -1, E)

        # Patch embed & pos embed
        x_s = embed(x_s, self.patch_embed_s, self.pos_embed_s, self.cls_token_s, self.unk_token_s)
        x_s = self.dropout(x_s)
        x_e = embed(x_e, self.patch_embed_e, self.pos_embed_e, self.cls_token_e, self.unk_token_e)
        x_e = self.dropout(x_e)

        # Encode
        x_s, x_e = self.encoder((x_s, x_e))

        # Classify
        x_s, x_e = x_s[:, 0], x_e[:, 0]
        x_s = self.mlp_s(x_s)
        x_e = self.mlp_e(x_e)
        x = x_s + x_e
        
        return x
