import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import einops
import torchvision.transforms.functional as ttf

from model.diffusion.modules import SinusoidalPosEmb

class CropRandomizer(nn.Module):
    """Randomly sample crops at input, and then average across crop features at output."""

    def __init__(
        self,
        input_shape,
        crop_height,
        crop_width,
        num_crops=1,
        pos_enc=False,
    ):
        """Args:
        input_shape (tuple, list): shape of input (not including batch dimension)
        crop_height (int): crop height
        crop_width (int): crop width
        num_crops (int): number of random crops to take
        pos_enc (bool): if True, add 2 channels to the output to encode the spatial
            location of the cropped pixels in the source image.
        """
        super().__init__()

        assert len(input_shape) == 3  # (C, H, W)
        assert crop_height <= input_shape[1]
        assert crop_width <= input_shape[2]

        self.input_shape = input_shape
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.num_crops = num_crops
        self.pos_enc = pos_enc

    def forward(self, inputs):
        """
        Args:
            inputs: (B, C, H, W) tensor
        Returns:
            Cropped tensor (B, C, crop_height, crop_width)
        """
        assert len(inputs.shape) == 4  # (B, C, H, W)
        
        if self.training:
            B, C, H, W = inputs.shape
            
            max_h = H - self.crop_height
            max_w = W - self.crop_width
            
            h_offsets = torch.randint(0, max_h + 1, (B,), device=inputs.device)
            w_offsets = torch.randint(0, max_w + 1, (B,), device=inputs.device)
            
            crops = []
            for i in range(B):
                crop = inputs[i:i+1, :, 
                            h_offsets[i]:h_offsets[i]+self.crop_height,
                            w_offsets[i]:w_offsets[i]+self.crop_width]
                crops.append(crop)
            
            return torch.cat(crops, dim=0)
        else:
            # Center crop during evaluation
            return ttf.center_crop(inputs, (self.crop_height, self.crop_width))

    def __repr__(self):
        """Pretty print network."""
        header = f"{str(self.__class__.__name__)}"
        msg = (
            header
            + f"(input_shape={self.input_shape}, crop_size=[{self.crop_height}, {self.crop_width}], num_crops={self.num_crops})"
        )
        return msg

class RobomimicSpatialSoftmax(nn.Module):
    """
    Spatial Softmax Layer adapted from robomimic.

    Input:  (B, C, H, W)
    Output: keypoints (B, num_kp, 2)
    """

    def __init__(
        self,
        input_shape,
        num_kp: int = 32,
        temperature: float = 1.0,
        learnable_temperature: bool = False,
        noise_std: float = 0.0,
    ):
        super().__init__()
        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape  # (C, H, W)

        if num_kp is not None:
            self.nets = nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._num_kp = num_kp
        else:
            self.nets = None
            self._num_kp = self._in_c

        self.learnable_temperature = learnable_temperature
        self.noise_std = noise_std

        if self.learnable_temperature:
            temperature = nn.Parameter(torch.ones(1) * temperature, requires_grad=True)
            self.register_parameter("temperature", temperature)
        else:
            temperature = nn.Parameter(torch.ones(1) * temperature, requires_grad=False)
            self.register_buffer("temperature", temperature)

        # coordinate grids in [-1, 1]
        pos_x, pos_y = np.meshgrid(
            np.linspace(-1.0, 1.0, self._in_w),
            np.linspace(-1.0, 1.0, self._in_h),
        )
        pos_x = torch.from_numpy(pos_x.reshape(1, self._in_h * self._in_w)).float()
        pos_y = torch.from_numpy(pos_y.reshape(1, self._in_h * self._in_w)).float()
        self.register_buffer("pos_x", pos_x)  # (1, H*W)
        self.register_buffer("pos_y", pos_y)  # (1, H*W)

        self.kps = None

    def forward(self, feature: torch.Tensor):
        """
        feature: (B, C, H, W)
        Returns:
            keypoints: (B, K, 2)
        """
        B, C, H, W = feature.shape
        assert C == self._in_c
        assert H == self._in_h
        assert W == self._in_w

        if self.nets is not None:
            feature = self.nets(feature)  # (B, K, H, W)
            K = self._num_kp
        else:
            K = C

        # [B, K, H, W] -> [B*K, H*W]
        feature = feature.reshape(B * K, H * W)

        # softmax over spatial locations
        attention = F.softmax(feature / self.temperature, dim=-1)  # (B*K, H*W)

        # expected x, y coordinates
        # pos_x, pos_y: (1, H*W)
        expected_x = torch.sum(self.pos_x * attention, dim=1, keepdim=True)  # (B*K, 1)
        expected_y = torch.sum(self.pos_y * attention, dim=1, keepdim=True)  # (B*K, 1)
        expected_xy = torch.cat([expected_x, expected_y], dim=1)             # (B*K, 2)

        # reshape to (B, K, 2)
        feature_keypoints = expected_xy.view(B, K, 2)

        if self.training and self.noise_std > 0.0:
            noise = torch.randn_like(feature_keypoints) * self.noise_std
            feature_keypoints = feature_keypoints + noise

        # store kps for debugging
        if isinstance(feature_keypoints, tuple):
            self.kps = (feature_keypoints[0].detach(), feature_keypoints[1].detach())
        else:
            self.kps = feature_keypoints.detach()

        return feature_keypoints


class ResNetSpatialEncoder(nn.Module):
    """
    One ResNet-18 encoder for a single camera view.

    Input:  (B, 3 * img_cond_steps, H, W)
    Output: (B, spatial_emb)
    """

    def __init__(
        self,
        input_channels: int,
        img_height: int,
        img_width: int,
        spatial_emb: int,
        num_kp: int = 32,
        softmax_temperature: float = 1.0,
        learnable_temperature: bool = False,
        noise_std: float = 0.0,
        head_dropout: float = 0.05,
        use_augmentation: bool = False,  # Whether to use random cropping
        crop_height: int = 84,  # Target crop height
        crop_width: int = 84,   # Target crop width
    ):
        super().__init__()

        # Create CropRandomizer for data augmentation if enabled
        if use_augmentation:
            input_shape = (input_channels, img_height, img_width)
            self.crop_randomizer = CropRandomizer(
                input_shape=input_shape,
                crop_height=crop_height,
                crop_width=crop_width,
                num_crops=1,
                pos_enc=False,
            )
            # Update dimensions for the cropped size
            img_height = crop_height
            img_width = crop_width
        else:
            self.crop_randomizer = None

        resnet = models.resnet18(weights=None)

        if input_channels != 3:
            resnet.conv1 = nn.Conv2d(
                input_channels,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )

        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

        # Infer feature shape for given resolution
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, img_height, img_width)
            feat = self.backbone(dummy)
            _, C, H, W = feat.shape

        self.feature_channels = C
        self.spatial_height = H
        self.spatial_width = W

        # robomimic-style spatial softmax
        self.spatial_softmax = RobomimicSpatialSoftmax(
            input_shape=(C, H, W),
            num_kp=num_kp,
            temperature=softmax_temperature,
            learnable_temperature=learnable_temperature,
            noise_std=noise_std,
        )

        # Map keypoints (K * 2) -> spatial_emb
        self.num_kp = num_kp
        self.head = nn.Sequential(
            nn.Linear(2 * num_kp, spatial_emb),
            nn.LayerNorm(spatial_emb),
            nn.ReLU(),
            nn.Dropout(head_dropout),
        )

        self.output_dim = spatial_emb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W) in [0,255] or [0,1]
        Returns: (B, spatial_emb)
        """
        # Apply cropping if enabled (random during training, center during eval)
        if self.crop_randomizer is not None:
            x = self.crop_randomizer(x)
        
        x = x / 255.0
        x = (x - 0.5) / 0.5  # map to roughly [-1, 1]

        feat = self.backbone(x)                # (B, C, H', W')
        kps = self.spatial_softmax(feat)       # (B, K, 2) or (kps, cov)

        if isinstance(kps, tuple):
            kps = kps[0]                       

        B, K, _ = kps.shape
        kps_flat = kps.view(B, K * 2)          # (B, 2K)

        out = self.head(kps_flat)              # (B, spatial_emb)
        return out


class ResBlock1D(nn.Module):
    """
    1D residual block with time + global condition modulation.
    x: (B, C, T)
    time_emb: (B, time_dim)
    cond_emb: (B, cond_dim)
    """

    def __init__(
        self,
        channels: int,
        time_dim: int,
        cond_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.channels = channels

        # Adaptive group norm to handle different channel sizes
        num_groups = min(8, channels)
        self.norm1 = nn.GroupNorm(num_groups, channels)
        self.norm2 = nn.GroupNorm(num_groups, channels)
        self.act = nn.SiLU()

        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)

        # time + cond -> scale, shift
        self.mlp = nn.Sequential(
            nn.Linear(time_dim + cond_dim, channels * 2),
            nn.SiLU(),
            nn.Linear(channels * 2, channels * 2),
        )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T)
        time_emb: (B, time_dim)
        cond_emb: (B, cond_dim)
        """
        B, C, T = x.shape

        # mod vector
        mod_input = torch.cat([time_emb, cond_emb], dim=-1)  # (B, time_dim + cond_dim)
        scale_shift = self.mlp(mod_input)                    # (B, 2C)
        scale, shift = scale_shift.chunk(2, dim=1)           # each (B, C)

        # reshape for broadcast over time
        scale = scale.unsqueeze(-1)        # (B, C, 1)
        shift = shift.unsqueeze(-1)        # (B, C, 1)

        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)

        # FiLM modulation
        h = h * (1 + scale) + shift

        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return x + h

class ActionUNet1D(nn.Module):
    """
    1D UNet over action horizon for flow matching.

    Input:
        x:            (B, Ta, Da)          - action trajectories
        time_emb:     (B, time_dim)        - time embedding
        fused_cond:   (B, fused_feature_dim) - fused [state + vision] features

    Output:
        (B, Ta, Da)
    """

    def __init__(
        self,
        action_dim: int,
        horizon_steps: int,
        time_dim: int,
        fused_feature_dim: int,
        base_channels: int = 128,
        channel_mults=(1, 2, 2),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.horizon_steps = horizon_steps
        self.time_dim = time_dim
        self.fused_feature_dim = fused_feature_dim

        # Channel schedule per resolution level
        self.channels = [base_channels * m for m in channel_mults]
        C0 = self.channels[0]

        # Project actions to first-level channels
        self.in_proj = nn.Conv1d(action_dim, C0, kernel_size=1)

        # ---------- Down path ----------
        self.down_resblocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()

        # Level 0
        self.down_resblocks.append(
            ResBlock1D(
                channels=C0,
                time_dim=time_dim,
                cond_dim=fused_feature_dim,   # <- use fused_feature_dim here
                dropout=dropout,
            )
        )

        # Levels 1..L-1
        for i in range(1, len(self.channels)):
            ch_in = self.channels[i - 1]
            ch_out = self.channels[i]
            self.down_samples.append(
                nn.Conv1d(ch_in, ch_out, kernel_size=4, stride=2, padding=1)
            )
            self.down_resblocks.append(
                ResBlock1D(
                    channels=ch_out,
                    time_dim=time_dim,
                    cond_dim=fused_feature_dim,
                    dropout=dropout,
                )
            )

        # ---------- Up path ----------
        self.up_samples = nn.ModuleList()
        self.up_resblocks = nn.ModuleList()
        for i in reversed(range(1, len(self.channels))):
            ch_in = self.channels[i]
            ch_out = self.channels[i - 1]
            self.up_samples.append(
                nn.ConvTranspose1d(
                    ch_in, ch_out, kernel_size=4, stride=2, padding=1
                )
            )
            self.up_resblocks.append(
                ResBlock1D(
                    channels=ch_out,
                    time_dim=time_dim,
                    cond_dim=fused_feature_dim,
                    dropout=dropout,
                )
            )

        # Final projection back to actions
        num_groups = min(8, C0)
        self.out_norm = nn.GroupNorm(num_groups, C0)
        self.out_act = nn.SiLU()
        self.out_proj = nn.Conv1d(C0, action_dim, kernel_size=1)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor, fused_cond: torch.Tensor) -> torch.Tensor:
        """
        x:         (B, Ta, Da)
        time_emb:  (B, time_dim)
        fused_cond:(B, fused_feature_dim)
        """
        B, Ta, Da = x.shape
        assert Ta == self.horizon_steps
        assert Da == self.action_dim

        # (B, Da, Ta)
        h = x.permute(0, 2, 1)
        h = self.in_proj(h)  # (B, C0, Ta)

        skips = []

        # ---------- Down path ----------
        # Level 0
        h = self.down_resblocks[0](h, time_emb, fused_cond)
        skips.append(h)

        # Levels 1..L-1
        for i in range(1, len(self.channels)):
            down = self.down_samples[i - 1]
            block = self.down_resblocks[i]

            h = down(h)
            h = block(h, time_emb, fused_cond)
            skips.append(h)

        # ---------- Up path ----------
        h = skips[-1]
        up_idx = 0

        for i in reversed(range(1, len(self.channels))):
            up = self.up_samples[up_idx]
            block = self.up_resblocks[up_idx]
            up_idx += 1

            h = up(h)
            skip = skips[i - 1]

            # Length align
            if h.shape[-1] != skip.shape[-1]:
                diff = skip.shape[-1] - h.shape[-1]
                if diff > 0:
                    h = F.pad(h, (0, diff))
                else:
                    h = h[:, :, : skip.shape[-1]]

            h = h + skip
            h = block(h, time_emb, fused_cond)

        h = self.out_norm(h)
        h = self.out_act(h)
        h = self.out_proj(h)  # (B, Da, Ta)
        return h.permute(0, 2, 1)  # (B, Ta, Da)

class VisionResNetFlowMatchingMLP(nn.Module):
    """
    Multi-view ResNet encoder + 1D UNet for flow matching policy over action horizon.

    Args:
        cond_dim:           flattened proprio state dimension (from cond["state"])
        fused_feature_dim:  dim of fused [state, vision] features used to condition the UNet
    """

    def __init__(
        self,
        action_dim: int,
        horizon_steps: int,
        cond_dim: int,
        img_cond_steps: int = 1,
        time_dim: int = 64,
        visual_feature_dim: int = 128,
        num_img: int = 2,
        img_height: int = 240,
        img_width: int = 240,
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
    ) -> None:
        super().__init__()

        self.num_img = num_img
        self.img_cond_steps = img_cond_steps
        self.spatial_emb = spatial_emb
        self.visual_feature_dim = visual_feature_dim
        self.cond_dim = cond_dim 
        self.fused_feature_dim = fused_feature_dim
        self.time_dim = time_dim
        self.action_dim = action_dim
        self.horizon_steps = horizon_steps

        # ---------- Vision encoders ----------
        self.encoders = nn.ModuleList()
        for _ in range(num_img):
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

        # Multi-view fusion: (B, num_img, spatial_emb) -> (B, visual_feature_dim)
        self.visual_proj = nn.Sequential(
            nn.Linear(num_img * spatial_emb, visual_feature_dim),
            nn.LayerNorm(visual_feature_dim),
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

        # ---------- Fused condition embedding (state + vision) ----------
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim + visual_feature_dim, fused_feature_dim),
            nn.LayerNorm(fused_feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
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

    def forward(self, x: torch.Tensor, time: torch.Tensor, cond: dict, **kwargs) -> torch.Tensor:
        """
        x: (B, Ta, Da) - action trajectories
        time: (B,) or int
        cond:
            state: (B, To, Do)
            rgb:   (B, To, C_all, H, W) with C_all = num_img * 3
        """
        B, Ta, Da = x.shape
        assert Ta == self.horizon_steps
        assert Da == self.action_dim

        # ---------- State ----------
        state = cond["state"].view(B, -1)

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
        out = self.unet(x, time_emb, fused_cond)                     # (B, Ta, Da)
        return out
    
    def extract_visual_features(self, cond: dict) -> torch.Tensor:
        """
        Extract visual features from RGB observations for storing in replay buffer.
        This method replicates the visual processing from forward() but only returns features.
        
        Args:
            cond: dict with key state/rgb
                state: (B, To, Do) - proprioceptive state
                rgb: (B, To, C, H, W) - visual observations
                
        Returns:
            visual_feat: (B, visual_feature_dim) - extracted visual features
        """
        B = cond["state"].shape[0]
        
        # Process images (same as in forward)
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
            time: (B,) or int, flow matching timestep
            state: (B, merged_feature_dim) or (B, cond_steps, merged_feature_dim) - 
                   merged features containing [low_dim_state, visual_features]
                   (Note: agent concatenates in this order to match forward() expectations)
            
        Returns:
            out: (B, Ta, Da) - predicted action trajectories
        """
        B, Ta, Da = x.shape
        assert Ta == self.horizon_steps
        assert Da == self.action_dim
        
        # Handle different state shapes
        if state.ndim == 3:
            # (B, cond_steps, merged_feature_dim) -> flatten
            state = state.reshape(B, -1)

        # Time embedding
        time = time.view(B, 1)
        time_emb = self.time_embedding(time).view(B, self.time_dim)
        
        fused_cond = self.cond_proj(state)  # (B, fused_feature_dim)
        
        # UNet over action horizon
        out = self.unet(x, time_emb, fused_cond)  # (B, Ta, Da)
        return out
