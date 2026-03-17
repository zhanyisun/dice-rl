"""
MLP models for diffusion policies.

"""

import torch
import torch.nn as nn
import logging
import einops
from copy import deepcopy

from model.common.mlp import MLP, ResidualMLP
from model.diffusion.modules import SinusoidalPosEmb
from model.common.modules import SpatialEmb, RandomShiftsAug

log = logging.getLogger(__name__)


class VisionDiffusionMLP(nn.Module):
    """With ViT backbone - original implementation for backward compatibility"""

    def __init__(
        self,
        backbone,
        action_dim,
        horizon_steps,
        cond_dim,
        img_cond_steps=1,
        time_dim=16,
        mlp_dims=[256, 256],
        activation_type="Mish",
        out_activation_type="Identity",
        use_layernorm=False,
        residual_style=False,
        spatial_emb=0,
        visual_feature_dim=128,
        dropout=0,
        num_img=1,
        augment=False,
    ):
        super().__init__()

        # vision
        self.backbone = backbone
        if augment:
            self.aug = RandomShiftsAug(pad=4)
        self.augment = augment
        self.num_img = num_img
        self.img_cond_steps = img_cond_steps
        if spatial_emb > 0:
            assert spatial_emb > 1, "this is the dimension"
            if num_img > 1:
                self.compress1 = SpatialEmb(
                    num_patch=self.backbone.num_patch,
                    patch_dim=self.backbone.patch_repr_dim,
                    prop_dim=cond_dim,
                    proj_dim=spatial_emb,
                    dropout=dropout,
                )
                self.compress2 = deepcopy(self.compress1)
            else:  # TODO: clean up
                self.compress = SpatialEmb(
                    num_patch=self.backbone.num_patch,
                    patch_dim=self.backbone.patch_repr_dim,
                    prop_dim=cond_dim,
                    proj_dim=spatial_emb,
                    dropout=dropout,
                )
            visual_feature_dim = spatial_emb * num_img
        else:
            self.compress = nn.Sequential(
                nn.Linear(self.backbone.repr_dim, visual_feature_dim),
                nn.LayerNorm(visual_feature_dim),
                nn.Dropout(dropout),
                nn.ReLU(),
            )

        # diffusion
        input_dim = (
            time_dim + action_dim * horizon_steps + visual_feature_dim + cond_dim
        )
        output_dim = action_dim * horizon_steps
        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.Mish(),
            nn.Linear(time_dim * 2, time_dim),
        )
        if residual_style:
            model = ResidualMLP
        else:
            model = MLP
        self.mlp_mean = model(
            [input_dim] + mlp_dims + [output_dim],
            activation_type=activation_type,
            out_activation_type=out_activation_type,
            use_layernorm=use_layernorm,
        )
        self.time_dim = time_dim

    def forward(
        self,
        x,
        time,
        cond: dict,
        **kwargs,
    ):
        """
        x: (B, Ta, Da)
        time: (B,) or int, diffusion step
        cond: dict with key state/rgb; more recent obs at the end
            state: (B, To, Do)
            rgb: (B, To, C, H, W)

        TODO long term: more flexible handling of cond
        """
        B, Ta, Da = x.shape
        _, T_rgb, C, H, W = cond["rgb"].shape

        # flatten chunk
        x = x.view(B, -1)

        # flatten history
        state = cond["state"].view(B, -1)

        # Take recent images --- sometimes we want to use fewer img_cond_steps than cond_steps (e.g., 1 image but 3 prio)
        rgb = cond["rgb"][:, -self.img_cond_steps :]

        # concatenate images in cond by channels
        if self.num_img > 1:
            rgb = rgb.reshape(B, T_rgb, self.num_img, 3, H, W)
            rgb = einops.rearrange(rgb, "b t n c h w -> b n (t c) h w")
        else:
            rgb = einops.rearrange(rgb, "b t c h w -> b (t c) h w")

        # convert rgb to float32 for augmentation
        rgb = rgb.float()

        # get vit output - pass in two images separately
        if self.num_img > 1:  # TODO: properly handle multiple images
            rgb1 = rgb[:, 0]
            rgb2 = rgb[:, 1]
            if self.augment:
                rgb1 = self.aug(rgb1)
                rgb2 = self.aug(rgb2)
            feat1 = self.backbone(rgb1)
            feat2 = self.backbone(rgb2)
            feat1 = self.compress1.forward(feat1, state)
            feat2 = self.compress2.forward(feat2, state)
            feat = torch.cat([feat1, feat2], dim=-1)
        else:  # single image
            if self.augment:
                rgb = self.aug(rgb)
            feat = self.backbone(rgb)

            # compress
            if isinstance(self.compress, SpatialEmb):
                feat = self.compress.forward(feat, state)
            else:
                feat = feat.flatten(1, -1)
                feat = self.compress(feat)
        cond_encoded = torch.cat([feat, state], dim=-1)

        # append time and cond
        time = time.view(B, 1)
        time_emb = self.time_embedding(time).view(B, self.time_dim)
        x = torch.cat([x, time_emb, cond_encoded], dim=-1)

        # mlp
        out = self.mlp_mean(x)
        return out.view(B, Ta, Da)


class TimeResidualBlockDiffusion(nn.Module):
    """Residual block modulated by timestep embeddings for DiffusionMLP."""

    def __init__(self, dim: int, hidden_dim: int, time_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        # output dimension should match the input "dim" so the addition works
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, dim))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        residual = x
        time_bias = self.time_mlp(time_emb)
        x = self.norm1(x)
        x = self.act(self.fc1(x + time_bias))
        x = self.dropout(x)
        x = self.act(self.fc2(x))
        x = self.dropout(x)
        x = x + residual
        x = self.norm2(x)
        return x


class DiffusionMLP(nn.Module):
    """MLP network for diffusion policy - matched to FlowMatchingMLP architecture."""

    def __init__(
        self,
        action_dim,
        horizon_steps,
        cond_dim,
        time_dim=16,
        mlp_dims=None,  # Deprecated, kept for compatibility
        hidden_dim=512,
        num_blocks=4,
        cond_mlp_dims=None,
        activation_type="Mish",  # Deprecated, kept for compatibility
        out_activation_type="Identity",  # Deprecated
        use_layernorm=False,  # Deprecated
        residual_style=False,  # Deprecated
        dropout=0.1,
    ):
        super().__init__()
        
        output_dim = action_dim * horizon_steps
        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.GELU(),
            nn.Linear(time_dim * 2, time_dim),
        )
        
        if cond_mlp_dims is not None:
            self.cond_mlp = MLP(
                [cond_dim] + cond_mlp_dims,
                activation_type="GELU",
                out_activation_type="Identity",
            )
            cond_out_dim = cond_mlp_dims[-1]
        else:
            cond_out_dim = cond_dim
            
        input_dim = time_dim + action_dim * horizon_steps + cond_out_dim
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [TimeResidualBlockDiffusion(hidden_dim, hidden_dim * 4, time_dim, dropout) for _ in range(num_blocks)]
        )
        self.final = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )
        nn.init.zeros_(self.final[-1].weight)
        nn.init.zeros_(self.final[-1].bias)
        self.time_dim = time_dim

    def forward(
        self,
        x,
        time,
        cond,
        **kwargs,
    ):
        """
        x: (B, Ta, Da)
        time: (B,) or int, diffusion step
        cond: dict with key state/rgb; more recent obs at the end
            state: (B, To, Do)
        """
        B, Ta, Da = x.shape
        x = x.view(B, -1)
        state = cond["state"].view(B, -1)
        if hasattr(self, "cond_mlp"):
            state = self.cond_mlp(state)

        time = time.view(B, 1)
        time_emb = self.time_embedding(time).view(B, self.time_dim)
        x = torch.cat([x, time_emb, state], dim=-1)

        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x, time_emb)
        out = self.final(x)
        return out.view(B, Ta, Da)

class ResidualBlock(nn.Module):
    """Simple residual block used by :class:`DiffusionMLPv2`."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.act = nn.GELU()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        x = self.act(self.fc2(x))
        x = self.dropout(x)
        x = residual + x
        x = self.norm2(x)
        return x


class DiffusionMLPv2(nn.Module):
    """Improved MLP diffusion model with residual blocks and dropout."""

    def __init__(
        self,
        action_dim: int,
        horizon_steps: int,
        cond_dim: int,
        time_dim: int = 16,
        hidden_dim: int = 512,
        num_blocks: int = 4,
        cond_mlp_dims=None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        output_dim = action_dim * horizon_steps
        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.GELU(),
            nn.Linear(time_dim * 2, time_dim),
        )

        if cond_mlp_dims is not None:
            self.cond_mlp = MLP(
                [cond_dim] + cond_mlp_dims,
                activation_type="GELU",
                out_activation_type="Identity",
            )
            cond_out_dim = cond_mlp_dims[-1]
        else:
            cond_out_dim = cond_dim

        input_dim = time_dim + action_dim * horizon_steps + cond_out_dim

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, hidden_dim * 4, dropout) for _ in range(num_blocks)]
        )
        self.final = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )
        self.time_dim = time_dim

    def forward(self, x: torch.Tensor, time: torch.Tensor, cond: dict, **kwargs) -> torch.Tensor:
        B, Ta, Da = x.shape
        x = x.view(B, -1)
        state = cond["state"].view(B, -1)
        if hasattr(self, "cond_mlp"):
            state = self.cond_mlp(state)

        time = time.view(B, 1)
        time_emb = self.time_embedding(time).view(B, self.time_dim)
        x = torch.cat([x, time_emb, state], dim=-1)

        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        out = self.final(x)
        return out.view(B, Ta, Da)