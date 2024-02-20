import torch
import torch.nn as nn
from einops import repeat
from typing import Literal


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
    def __init__(self, embed_dim, num_heads, dropout, cheap_cross=False):
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
        self.cross_in_s = nn.Linear(embed_dim, embed_dim)
        self.cross_out_s = nn.Linear(embed_dim, embed_dim)
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
        self.cross_in_e = nn.Linear(embed_dim, embed_dim)
        self.cross_out_e = nn.Linear(embed_dim, embed_dim)
        self.mlp_e = Mlp(embed_dim, embed_dim, embed_dim, dropout)

        self.norm_e_1 = nn.LayerNorm(embed_dim)
        self.norm_e_2 = nn.LayerNorm(embed_dim)
        self.norm_e_3 = nn.LayerNorm(embed_dim)

        # Common
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.cheap_cross = cheap_cross
    
    def forward(self, x):
        x_s, x_e = x
        
        # Self
        if x_s is not None:
            x_s = self.norm_s_1(x_s + self.dropout(self.self_attn_s(x_s, x_s, x_s, need_weights=False)[0]))
        if x_e is not None:
            x_e = self.norm_e_1(x_e + self.dropout(self.self_attn_e(x_e, x_e, x_e, need_weights=False)[0]))

        # Cross
        if x_s is not None and x_e is not None:
            if self.cheap_cross:
                # Note: norm_*_2 not used
                class_token_s, x_s = x_s[:, 0:1, :], x_s[:, 1:, :]
                class_token_e, x_e = x_e[:, 0:1, :], x_e[:, 1:, :]

                class_token_s = self.cross_in_s(class_token_s)
                class_token_s = class_token_s + self.cross_attn_s(class_token_s, x_e, x_e, need_weights=False)[0]
                class_token_s = self.cross_out_s(class_token_s)
                x_s_after_cross = torch.cat([class_token_s, x_s], dim=1)
                
                class_token_e = self.cross_in_e(class_token_e)
                class_token_e = class_token_e + self.cross_attn_e(class_token_e, x_s, x_s, need_weights=False)[0]
                class_token_e = self.cross_out_e(class_token_e)
                x_e_after_cross = torch.cat([class_token_e, x_e], dim=1)
            else:
                x_s_after_cross = self.norm_s_2(x_s + self.dropout(self.cross_attn_s(x_s, x_e, x_e, need_weights=False)[0]))
                x_e_after_cross = self.norm_e_2(x_e + self.dropout(self.cross_attn_e(x_e, x_s, x_s, need_weights=False)[0]))
            x_s, x_e = x_s_after_cross, x_e_after_cross
        else:
            x_s_after_cross, x_e_after_cross = x_s, x_e

        # Mlp
        if x_s is not None:
            x_s = self.norm_s_3(x_s + self.mlp_s(x_s))
        if x_e is not None:
            x_e = self.norm_e_3(x_e + self.mlp_e(x_e))

        x = x_s, x_e

        return x


def embed(x, patch_embed, pos_embed, cls_token, nan_token):
    B, N, _ = x.shape

    # Flatten to B * N to ease masking
    x = x.reshape(B * N, -1)

    # Fill nans
    # Note: here nan token is introduced before patch embed
    # to avoid messing with NaNs when applying linear layer
    mask = x.isnan().any(1)
    x[mask] = nan_token.expand(mask.sum(), -1)

    # Reshape back
    x = x.reshape(B, N, -1)

    # Patch embed
    x = patch_embed(x)

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
        cheap_cross,
        pool: Literal['cls', '10sec'] = 'cls',
    ):
        super().__init__()

        # Patch embedding
        self.patch_embed_s = nn.Linear(input_dim_s, embed_dim)
        self.patch_embed_e = nn.Linear(input_dim_e, embed_dim)

        # Pos embedding
        self.pos_embed_s = nn.Parameter(torch.randn(1, num_patches_s + 1, embed_dim))
        self.cls_token_s = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.nan_token_s = nn.Parameter(torch.randn(input_dim_s))

        self.pos_embed_e = nn.Parameter(torch.randn(1, num_patches_e + 1, embed_dim))
        self.cls_token_e = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.nan_token_e = nn.Parameter(torch.randn(input_dim_e))

        # Common
        self.dropout = nn.Dropout(dropout)
        self.pool = pool

        # Encoder
        self.encoder = nn.Sequential(
            *[
                HmsEncoderLayer(embed_dim, num_heads, dropout, cheap_cross)
                for _ in range(depth)
            ]
        )

        # Mlps
        self.mlp_s = Mlp(embed_dim, embed_dim, n_classes, dropout)
        self.mlp_e = Mlp(embed_dim, embed_dim, n_classes, dropout)
    
    def forward(self, x_s, x_e):
        assert x_s is not None or x_e is not None

        # Flatten patches
        if x_s is not None:
            # s: (B, K=4, T=300, E=100) -> (B, N=1200, E=100)
            B_s, K_s, T_s, E_s = x_s.shape
            x_s = x_s.reshape(B_s, -1, E_s)

            # Patch embed & pos embed
            x_s = embed(x_s, self.patch_embed_s, self.pos_embed_s, self.cls_token_s, self.nan_token_s)
            x_s = self.dropout(x_s)
        
        if x_e is not None:
            # e: (B, T=50, E=200, F=20) -> (B, N=1000, E=200)
            B_e, T_e, E_e, F_e = x_e.shape
            x_e = x_e.permute([0, 1, 3, 2])
            x_e = x_e.reshape(B_e, -1, E_e)

            # Patch embed & pos embed
            x_e = embed(x_e, self.patch_embed_e, self.pos_embed_e, self.cls_token_e, self.nan_token_e)
            x_e = self.dropout(x_e)

        # Encode
        x_s, x_e = self.encoder((x_s, x_e))

        # Classify
        x = 0
        if x_s is not None:
            if self.pool == 'cls':
                # Use single class token
                x_s = x_s[:, 0]  # (B, E')
            elif self.pool == '10sec':
                # TODO: experiment with alignment (-3 +2 or -2 +3)
                # Select "middle" 10 sec (T = 300 == 600 sec)
                # x_s: (B, N = K * T = 4 * 300 = 1200, E')
                x_s = x_s[:, 1:, :]
                x_s = x_s.reshape(B_s, K_s, T_s, -1)
                T_center_index = T_s // 2
                x_s = x_s[:, :, T_center_index - 3:T_center_index + 2, :].mean((1, 2))  # (B, E')
            else:
                raise ValueError(f'unknown pool {self.pool}')
            x_s = self.mlp_s(x_s)
            x = x + x_s / 2
        if x_e is not None:
            if self.pool == 'cls':
                # Use single class token
                x_e = x_e[:, 0]  # (B, E')
            elif self.pool == '10sec':
                # Select middle 10 sec (T = 50 == 50 sec)
                # x_e: (B, N = T * F = 50 * 20 = 1000, E')
                x_e = x_e[:, 1:, :]
                x_e = x_e.reshape(B_e, T_e, F_e, -1)
                T_center_index = T_e // 2
                x_e = x_e[:, T_center_index - 5:T_center_index + 5, :, :].mean((1, 2))  # (B, E')
            else:
                raise ValueError(f'unknown pool {self.pool}')
            x_e = self.mlp_e(x_e)
            x = x + x_e / 2
        
        return x
