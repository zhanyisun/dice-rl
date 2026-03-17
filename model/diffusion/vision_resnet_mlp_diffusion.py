"""ResNet-based vision diffusion MLP models matching flow matching architecture."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import einops

from model.diffusion.modules import SinusoidalPosEmb

# Import components from flow matching
try:
    from model.flow_matching.vision_resnet_mlp_flow_matching import (
        ResNetSpatialEncoder, 
        CropRandomizer,
        RobomimicSpatialSoftmax,
        ActionUNet1D,
        ResBlock1D
    )
except ImportError:
    # Define fallbacks if imports fail
    ResNetSpatialEncoder = None
    CropRandomizer = None
    RobomimicSpatialSoftmax = None
    ActionUNet1D = None
    ResBlock1D = None


class TimeResidualBlock(nn.Module):
    """Residual block modulated by timestep embeddings for diffusion."""

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


class VisionResNetDiffusionMLP(nn.Module):
    """
    Multi-view ResNet + 1D UNet for diffusion policy over action horizon.
    Matches the architecture of VisionResNetFlowMatchingMLP for fair comparison.
    
    Args:
        cond_dim:           flattened proprio state dimension (from cond["state"])
        fused_feature_dim:  dim of fused [state, vision] features used to condition the UNet
    """

    def __init__(
        self,
        action_dim: int,
        horizon_steps: int,
        cond_dim: int,                  # flattened state dim
        img_cond_steps: int = 1,
        time_dim: int = 64,
        visual_feature_dim: int = 128,  
        num_img: int = 2,
        img_height: int = 96,
        img_width: int = 96,
        spatial_emb: int = 64,
        num_kp: int = 32,
        fused_feature_dim: int = 128,
        unet_base_channels: int = 128,
        unet_channel_mults=(1, 2, 2),
        dropout: float = 0.1,
        # Augmentation params (for cropping)
        use_augmentation: bool = False,
        crop_height: int = 84,
        crop_width: int = 84,
    ):
        super().__init__()
        
        self.num_img = num_img
        self.img_cond_steps = img_cond_steps
        self.spatial_emb = spatial_emb
        self.visual_feature_dim = visual_feature_dim
        self.cond_dim = cond_dim  # flattened state dim
        self.fused_feature_dim = fused_feature_dim
        self.time_dim = time_dim
        self.action_dim = action_dim
        self.horizon_steps = horizon_steps
        
        # ---------- Vision encoders ----------
        self.encoders = nn.ModuleList()
        for _ in range(num_img):
            if ResNetSpatialEncoder is not None:
                encoder = ResNetSpatialEncoder(
                    input_channels=3 * img_cond_steps,
                    img_height=img_height,
                    img_width=img_width,
                    spatial_emb=spatial_emb,
                    num_kp=num_kp,
                    softmax_temperature=1.0,
                    learnable_temperature=False,
                    noise_std=0.0,
                    use_augmentation=use_augmentation,
                    crop_height=crop_height,
                    crop_width=crop_width,
                )
                self.encoders.append(encoder)
            else:
                raise ImportError("Cannot import ResNetSpatialEncoder from flow matching module")
        
        # Multi-view fusion: (B, num_img, spatial_emb) -> (B, visual_feature_dim)
        self.visual_proj = nn.Sequential(
            nn.Linear(num_img * spatial_emb, visual_feature_dim),
            nn.LayerNorm(visual_feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # ---------- Fused condition embedding (state + vision) ----------
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim + visual_feature_dim, fused_feature_dim),
            nn.LayerNorm(fused_feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # ---------- Time embedding ----------
        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.GELU(),
            nn.Linear(time_dim * 2, time_dim),
        )
        
        # ---------- 1D UNet over action horizon ----------
        self.unet = ActionUNet1D(
            action_dim=action_dim,
            horizon_steps=horizon_steps,
            time_dim=time_dim,
            fused_feature_dim=fused_feature_dim,
            base_channels=unet_base_channels,
            channel_mults=unet_channel_mults,
            dropout=dropout,
        )
        
    def extract_visual_features(self, cond: dict) -> torch.Tensor:
        """
        Extract visual features from RGB observations for storing in replay buffer.
        
        Args:
            cond: dict with key state/rgb
                state: (B, To, Do) - proprioceptive state
                rgb: (B, To, C, H, W) - visual observations
                
        Returns:
            visual_feat: (B, visual_feature_dim) - extracted visual features
        """
        B = cond["state"].shape[0]
        
        # ---------- RGB ----------
        rgb = cond["rgb"][:, -self.img_cond_steps:]
        B_rgb, T_img, C_all, H, W = rgb.shape
        assert B_rgb == B
        assert T_img == self.img_cond_steps
        assert C_all == self.num_img * 3
        channels_per_cam = 3
        
        rgb = rgb.reshape(
            B,
            T_img,
            self.num_img,
            channels_per_cam,
            H,
            W,
        )
        rgb = einops.rearrange(rgb, "b t n c h w -> b n (t c) h w")
        # (B, num_img, 3 * img_cond_steps, H, W)
        
        cam_feats = []
        for i in range(self.num_img):
            cam_rgb = rgb[:, i]
            cam_feat = self.encoders[i](cam_rgb)  # (B, spatial_emb)
            cam_feats.append(cam_feat)
        
        cam_feats = torch.stack(cam_feats, dim=1)     # (B, num_img, spatial_emb)
        cam_feats_flat = cam_feats.reshape(B, -1)     # (B, num_img * spatial_emb)
        visual_feat = self.visual_proj(cam_feats_flat)  # (B, visual_feature_dim)
        
        return visual_feat
    
    def forward_from_features(self, x: torch.Tensor, time: torch.Tensor, state: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass using merged state (visual features + low-dim state).
        This is used during online RL when the environment wrapper merges visual features with state.
        
        Args:
            x: (B, Ta, Da) - action trajectories
            time: (B,) or int, diffusion timestep
            state: (B, merged_feature_dim) or (B, cond_steps, merged_feature_dim) - 
                   merged features containing [low_dim_state, visual_features]
            
        Returns:
            out: (B, Ta, Da) - predicted noise/actions
        """
        B, Ta, Da = x.shape
        assert Ta == self.horizon_steps
        assert Da == self.action_dim
        
        # Handle different state shapes
        if state.ndim == 3:
            # (B, cond_steps, merged_feature_dim) -> flatten
            state = state.reshape(B, -1)
        
        # The merged state already contains [low_dim_state, visual_features] in the right order
        # Just pass it directly to cond_proj which expects this concatenation
        
        # Time embedding
        time = time.view(B, 1)
        time_emb = self.time_embedding(time).view(B, self.time_dim)  # (B, time_dim)
        
        # Fused condition - state already has the right order [low_dim_state, visual_features]
        fused_cond = self.cond_proj(state)  # (B, fused_feature_dim)
        
        # UNet over action horizon
        if hasattr(self, 'unet'):
            out = self.unet(x, time_emb, fused_cond)  # (B, Ta, Da)
        else:
            # Fallback to MLP
            x_flat = x.view(B, -1)
            x_input = torch.cat([x_flat, time_emb, fused_cond], dim=-1)
            
            h = self.input_proj(x_input)
            for block in self.blocks:
                h = block(h, time_emb)
            out = self.final(h)
            out = out.view(B, Ta, Da)
        
        return out
    
    def forward(self, x: torch.Tensor, time: torch.Tensor, cond: dict, **kwargs) -> torch.Tensor:
        """
        Forward pass with full observation processing.
        
        Args:
            x: (B, Ta, Da) - action trajectories  
            time: (B,) or int, diffusion timestep
            cond: dict with key state/rgb
                state: (B, To, Do) - proprioceptive state
                rgb: (B, To, C, H, W) - visual observations
                
        Returns:
            out: (B, Ta, Da) - predicted noise/actions
        """
        B, Ta, Da = x.shape
        assert Ta == self.horizon_steps
        assert Da == self.action_dim
        
        # ---------- State ----------
        state = cond["state"].view(B, -1)
        # Note: state.shape[-1] could be cond_dim * cond_steps if multiple timesteps are provided
        # We just use the flattened version as-is
        
        # ---------- RGB ----------
        rgb = cond["rgb"][:, -self.img_cond_steps:]
        B_rgb, T_img, C_all, H, W = rgb.shape
        assert B_rgb == B
        assert T_img == self.img_cond_steps
        assert C_all == self.num_img * 3
        channels_per_cam = 3
        
        rgb = rgb.reshape(
            B,
            T_img,
            self.num_img,
            channels_per_cam,
            H,
            W,
        )
        rgb = einops.rearrange(rgb, "b t n c h w -> b n (t c) h w")
        # (B, num_img, 3 * img_cond_steps, H, W)
        
        cam_feats = []
        for i in range(self.num_img):
            cam_rgb = rgb[:, i]
            cam_feat = self.encoders[i](cam_rgb)  # (B, spatial_emb)
            cam_feats.append(cam_feat)
        
        cam_feats = torch.stack(cam_feats, dim=1)     # (B, num_img, spatial_emb)
        cam_feats_flat = cam_feats.reshape(B, -1)     # (B, num_img * spatial_emb)
        visual_feat = self.visual_proj(cam_feats_flat)  # (B, visual_feature_dim)
        
        # ---------- Time embedding ----------
        time = time.view(B, 1)
        time_emb = self.time_embedding(time).view(B, self.time_dim)  # (B, time_dim)
        
        # ---------- Fused condition ----------
        fused_input = torch.cat([state, visual_feat], dim=-1)        # (B, cond_dim + visual_feature_dim)
        fused_cond = self.cond_proj(fused_input)                     # (B, fused_feature_dim)
        
        # ---------- UNet over action horizon ----------
        if hasattr(self, 'unet'):
            out = self.unet(x, time_emb, fused_cond)                 # (B, Ta, Da)
        else:
            # Fallback to MLP
            x_flat = x.view(B, -1)
            x_input = torch.cat([x_flat, time_emb, fused_cond], dim=-1)
            
            h = self.input_proj(x_input)
            for block in self.blocks:
                h = block(h, time_emb)
            out = self.final(h)
            out = out.view(B, Ta, Da)
        
        return out