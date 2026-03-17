"""
Distilled RL Model for image-based online finetuning.

This module extends DistillRLModel to handle image observations with merged visual features.
The key difference is that the state already contains visual features merged by the environment wrapper.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import hydra
import os
from omegaconf import OmegaConf

log = logging.getLogger(__name__)

from model.rl.distill_residual_rl import DistillResidualRLModel


class DistillResidualRLImgModel(DistillResidualRLModel):
    """
    Image-based distilled RL model for online finetuning.
    
    This model works with augmented state dimensions where visual features
    are already merged with low-dim state by the environment wrapper.
    
    The main differences from DistillRLModel:
    1. get_Action uses forward_from_features
    2. pretrain_on_expert_data handles image data and extracts features
    3. _load_pretrained_policy loads VisionFlowMatchingMLP
    """
    
    def __init__(
        self,
        obs_dim: int,  # This should be augmented (visual_feature_dim + original_obs_dim)
        action_dim: int,
        pretrained_flow_policy_path: str,
        # All other parameters with defaults to match parent
        **kwargs
    ):
        """
        Initialize image-based distilled RL model.
        
        Args:
            obs_dim: Augmented observation dimension (visual_feature_dim + original_obs_dim)
            action_dim: Action dimension
            pretrained_flow_policy_path: Path to pretrained flow policy
            **kwargs: All other parameters passed to parent class
        """
        # Store augmented obs_dim for logging
        self.augmented_obs_dim = obs_dim
        # Call parent init with all parameters in correct order
        super().__init__(
            obs_dim=obs_dim,
            action_dim=action_dim,
            pretrained_flow_policy_path=pretrained_flow_policy_path,
            **kwargs
        )

        log.info(f"DistillResidualRLImgModel initialized with augmented obs_dim={obs_dim}")
    
    def _load_pretrained_policy(self, checkpoint_path: str, device: str):
        """
        Load pretrained VisionFlowMatchingMLP policy.
        
        Override parent's method to verify it's an image-based model.
        
        Args:
            checkpoint_path: Path to the pretrained policy checkpoint (.pt file)
            device: Device to load the model on
            
        Returns:
            Loaded and frozen VisionFlowMatchingMLP
        """
        # Use parent's loading logic
        pretrained_model = super()._load_pretrained_policy(checkpoint_path, device)
        
        # Verify it's an image-based model with required methods
        if not hasattr(pretrained_model, 'network') or not hasattr(pretrained_model.network, 'forward_from_features'):
            log.warning("Pretrained model doesn't have expected structure for image-based model")
        
        log.info("Pretrained image-based flow matching policy loaded and frozen successfully")
        return pretrained_model
    
    def get_action(self, state: torch.Tensor, noise: torch.Tensor, return_pretrained_actions: bool = False):
        """
        Get action as sum of pretrained policy and residual actor.
        
        Action = π_pre(s,z) + r_θ(s,z) or π_pre(s,z) + r_θ(s,a_base)
        
        Args:
            state: (B, cond_steps, augmented_obs_dim) - state with visual features already merged
            noise: (B, horizon_steps, action_dim) - noise for action generation
            return_pretrained_actions: if True, return tuple (total_actions, pretrained_actions)
            
        Returns:
            if return_pretrained_actions:
                tuple: (total_actions, pretrained_actions) both (B, horizon_steps, action_dim)
            else:
                action: (B, horizon_steps, action_dim) - total action (pretrained + residual)
        """
        # Get pretrained action (no gradient)
        with torch.no_grad():
            # The state already contains merged visual features from the agent
            # Use forward_from_features for image-based model
            output = self.pretrained_flow_policy.forward_from_features(
                features=state,
                init_noise=noise
            )
            pretrained_actions = output.trajectories  # (B, horizon_steps, action_dim)
            pretrained_actions = pretrained_actions.detach()
        
        # Get residual action from actor - condition on base action or noise
        if self.condition_residual_on_base_action:
            # Use base action as input: r_θ(s, a_base)
            residual_actions = self.actor(state, pretrained_actions)  # (B, horizon_steps, action_dim)
        else:
            # Use noise as input: r_θ(s, z) 
            residual_actions = self.actor(state, noise)  # (B, horizon_steps, action_dim)
        
        # Total action = pretrained + residual
        total_actions = pretrained_actions + residual_actions
        
        if return_pretrained_actions:
            return total_actions, pretrained_actions
        else:
            return total_actions

    def get_exploration_action(self, state: torch.Tensor, num_samples: int = 10,
                              exploration_strategy: str = "max_q_std", training_step: int = 0,
                              replay_flow_model=None, replay_flow_config=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get exploration action using specified strategy.
        
        Image-based version that works with augmented states.
        Supports two strategies:
        - max_q_std: Select action with highest Q-std across ensemble (epistemic uncertainty)
        - max_q_min: Select action with highest minimum Q-value across ensemble (optimistic)
        
        If training_step < replay_flow_warmup_steps, uses single sample regardless of strategy.
        
        Args:
            state: (B, cond_steps, augmented_obs_dim) - state with visual features already merged
            num_samples: Number of noise samples to evaluate
            exploration_strategy: "max_q_std" or "max_q_min"
            training_step: Current training step (to check against warmup)
            
        Returns:
            selected_action: (B, horizon_steps, action_dim) - selected action
            selected_noise: (B, horizon_steps, action_dim) - corresponding noise
        """
        B = state.shape[0]
        device = state.device
        
        # During warmup, use single sample regardless of strategy
        if training_step <= self.replay_flow_warmup_steps:
            # Single sample during warmup
            noise = torch.randn(B, self.horizon_steps, self.action_dim, device=device)
            action = self.get_action(state, noise)
            return action, noise
        
        # After warmup, use specified exploration strategy
        # Sample multiple noise vectors
        noise_samples = torch.randn(num_samples, B, self.horizon_steps, self.action_dim, device=device)
        
        # Expand state for batch processing
        state_expanded = state.unsqueeze(0).expand(num_samples, -1, -1, -1)
        state_flat = state_expanded.reshape(num_samples * B, *state.shape[1:])
        noise_flat = noise_samples.reshape(num_samples * B, self.horizon_steps, self.action_dim)
        
        # Get actions for all noise samples
        with torch.no_grad():
            actions_flat = self.get_action(state_flat, noise_flat)  # (num_samples * B, horizon_steps, action_dim)
            
            # Get Q-values from all critics for all samples
            q_all = self.critic(state_flat, noise_flat, actions_flat, return_all=True)  # List of (num_samples * B, 1)
            
            # Stack Q-values from ensemble: (ensemble_size, num_samples * B, 1)
            q_stacked = torch.stack(q_all, dim=0)
            
            # Reshape to separate samples: (ensemble_size, num_samples, B, 1)
            q_reshaped = q_stacked.view(len(q_all), num_samples, B, 1)
            
            if exploration_strategy == "max_q_min":
                # Select action with max of min Q-value across ensemble
                # min over ensemble for each sample: (num_samples, B, 1)
                q_min = q_reshaped.min(dim=0)[0]
                
                # Select sample with max min Q for each batch element
                max_min_indices = q_min.squeeze(-1).argmax(dim=0)  # (B,)
                selection_indices = max_min_indices

            elif exploration_strategy == "max_q_std":  # max_q_std (default)
                # Select action with max Q-std across ensemble
                # Compute std across ensemble for each sample: (num_samples, B, 1)
                q_std = q_reshaped.std(dim=0)
                
                # Select sample with max std for each batch element
                max_std_indices = q_std.squeeze(-1).argmax(dim=0)  # (B,)
                selection_indices = max_std_indices
            elif exploration_strategy == "max_q_std_filtered_by_min":
                # Select top 3 samples based on min q value, then from those select max std
                # Get min Q-value across ensemble for each sample: (num_samples, B, 1)
                q_min = q_reshaped.min(dim=0)[0]
                
                # Get top 3 samples for each batch element based on min Q
                top_k = min(3, num_samples)  # In case we have fewer than 3 samples
                top_q_values, top_indices = q_min.squeeze(-1).topk(top_k, dim=0)  # (top_k, B)
                
                # Compute std for the top-k samples only
                # Extract Q-values for top-k samples: (ensemble_size, top_k, B, 1)
                top_q_reshaped = torch.stack([
                    q_reshaped[:, top_indices[k], torch.arange(B)]  # (ensemble_size, B, 1)
                    for k in range(top_k)
                ], dim=1)  # (ensemble_size, top_k, B, 1)
                
                # Compute std across ensemble for top-k samples: (top_k, B, 1)
                top_q_std = top_q_reshaped.std(dim=0)
                
                # Select the sample with max std from the top-k
                max_std_in_topk_indices = top_q_std.squeeze(-1).argmax(dim=0)  # (B,)
                
                # Get the actual sample indices from the top-k indices
                selection_indices = torch.gather(top_indices, 0, max_std_in_topk_indices.unsqueeze(0)).squeeze(0)  # (B,)
            

            actions_reshaped = actions_flat.view(num_samples, B, self.horizon_steps, self.action_dim)
            

            selected_actions = torch.stack([
                actions_reshaped[selection_indices[b], b] for b in range(B)
            ])
            
            # Select corresponding noise
            selected_noise = torch.stack([
                noise_samples[selection_indices[b], b] for b in range(B)
            ])
        
        return selected_actions, selected_noise
    
    def actor_loss(
        self,
        state: torch.Tensor,
        noise: torch.Tensor,
        current_actions: torch.Tensor,  # (B, H, A) - total actions (pretrained + residual)
        q_values: torch.Tensor,         # (B, 1)
        next_state: Optional[torch.Tensor] = None,
        next_noise: Optional[torch.Tensor] = None,
        training_step: int = 0,
        q_overestimation: Optional[torch.Tensor] = None,  # (B,1) if provided
        data_source: Optional[torch.Tensor] = None,  # (B,1) - 0 for online, 1 for expert
    ) -> Dict[str, torch.Tensor]:
        """
        Actor loss for residual RL with multi-z sampling (always assumes sample_multi_z_for_actor_loss=True).
        
        Different behavior based on training_step vs q_filtering_warmup_steps:
        1. During warmup (≤ q_filtering_warmup_steps): Simple Q + BC loss, no filtering
        2. After warmup: BC filtering and self-imitation when conditions are met
        """
        B = state.shape[0]
        K = self.num_multi_z_for_actor_loss
        
        # Sample K noise vectors for each state
        noise_samples = torch.randn(B, K, *noise.shape[1:], device=self.device)  # (B, K, H, A)
        
        # Compute actions for all K noise samples
        state_expanded = state.unsqueeze(1).expand(-1, K, -1, -1)  # (B, K, cond_steps, obs_dim)
        state_flat = state_expanded.reshape(B * K, *state.shape[1:])  # (B*K, cond_steps, obs_dim)
        noise_flat = noise_samples.reshape(B * K, *noise_samples.shape[2:])  # (B*K, H, A) - use noise_samples shape, not noise shape!
        
        # Get actions with pretrained actions returned
        actions_flat, pretrained_actions_flat = self.get_action(state_flat, noise_flat, return_pretrained_actions=True)  # (B*K, H, A)
        
        # Reshape back to (B, K, H, A)
        actions_samples = actions_flat.reshape(B, K, *actions_flat.shape[1:])  # (B, K, H, A)
        pretrained_actions_samples = pretrained_actions_flat.reshape(B, K, *pretrained_actions_flat.shape[1:])  # (B, K, H, A)
        
        # Compute Q-values for current actions (with gradients for actor loss)
        q_current_flat = self.critic(state_flat, noise_flat, actions_flat)  # (B*K, 1)
        q_current_samples = q_current_flat.reshape(B, K)  # (B, K)
        
        # Compute Q-values for pretrained actions (no gradients)
        with torch.no_grad():
            q_pretrained_flat = self.critic(state_flat, noise_flat, pretrained_actions_flat)  # (B*K, 1)
            q_pretrained_samples = q_pretrained_flat.reshape(B, K)  # (B, K)
        
        # Check if we're in warmup phase
        in_warmup = training_step <= self.q_filtering_warmup_steps
        
        if in_warmup:
            # Q-value loss: -mean(Q(s, a^current_k))
            q_loss_per_batch = q_current_samples.mean(dim=1)  # (B,) - mean over K samples
            
            # Apply disable_q_loss_for_expert_data if enabled
            online_mask = None
            if self.disable_q_loss_for_expert_data and data_source is not None:
                # Mask Q loss for expert data: only apply Q loss to online data (data_source == 0)
                online_mask = (data_source == 0).float().squeeze(-1)  # (B, 1) -> (B,) - 1.0 for online, 0.0 for expert
                q_loss_per_batch = q_loss_per_batch * online_mask  # (B,)
            
            q_loss = -q_loss_per_batch.mean()  # scalar
            
            # Normalize Q-loss by mean absolute Q-value for stability (like FQL)
            if self.use_q_normalization:
                # Compute normalization constant only from the Q-values we're actually using for loss
                if online_mask is not None:
                    # Only normalize by online Q-values when we're masking expert data
                    online_mask_expanded = online_mask.unsqueeze(-1).expand(-1, K)  # (B, K)
                    online_q_values = q_current_samples * online_mask_expanded  # Zero out expert Q-values
                    online_count = online_mask_expanded.sum()  # Count of online samples
                    if online_count > 0:
                        q_abs_mean = (online_q_values.abs().sum() / online_count).detach()  # Mean of online Q-values only
                    else:
                        q_abs_mean = q_current_samples.abs().mean().detach()  # Fallback to all if no online samples
                else:
                    # Use all Q-values for normalization
                    q_abs_mean = q_current_samples.abs().mean().detach()
                
                if q_abs_mean > 1e-8:  # Avoid division by zero
                    q_loss_scale = 1.0 / q_abs_mean
                    q_loss = q_loss_scale * q_loss
                
            # BC regularization loss: mean(||a^current_k - a^pre_k||²)
            action_diff = actions_samples - pretrained_actions_samples  # (B, K, horizon_steps, action_dim)
            # Compute MSE across action dimensions for each timestep, then average across time
            mse_per_timestep = (action_diff ** 2).mean(dim=-1)  # (B, K, horizon_steps)
            # Average across timesteps to get per-sample MSE: (B, K)
            mse_per_sample = mse_per_timestep.mean(dim=-1)  # (B, K)
            # Mean across K samples and average across batch
            filtered_bc_loss = mse_per_sample.mean(dim=1).mean()  # scalar
            
            # Total loss for warmup
            total_loss = q_loss + self.bc_loss_weight * filtered_bc_loss
            
            # Set placeholders for metrics
            bc_filter = torch.ones(B, 1, device=self.device)
            better_percentage = torch.tensor(0.0, device=self.device)
            q_advantage = torch.zeros(B, 1, device=self.device)
            avg_q_current = q_current_samples.mean(dim=1, keepdim=True)
            avg_q_pretrained = q_pretrained_samples.mean(dim=1, keepdim=True)
            
        else:
            # POST-WARMUP PHASE: With BC filtering and self-imitation
            
            # Compute per-sample Q-advantages for filtering (no gradients needed for filtering)
            with torch.no_grad():
                # Per-sample Q-advantage: q_current_bk - q_pretrained_bk for each k
                q_advantage_per_sample = q_current_samples - q_pretrained_samples  # (B, K)
                better_than_expert_per_sample = (q_advantage_per_sample > 0).float()  # (B, K) - 1.0 if better, 0.0 if worse
                
                # Compute average metrics for logging (keep these for backward compatibility)
                avg_q_current = q_current_samples.mean(dim=1, keepdim=True)  # (B, 1)
                avg_q_pretrained = q_pretrained_samples.mean(dim=1, keepdim=True)  # (B, 1)
                q_advantage = avg_q_current - avg_q_pretrained  # (B, 1) - average advantage for logging
                better_than_expert = (q_advantage > 0).float()  # (B, 1) - average for logging
                better_percentage = better_than_expert_per_sample.mean()  # Fraction of all (B,K) samples where policy is better
                
                # Apply soft Q-filtering on per-sample basis
                if self.use_soft_q_filtering:
                    # Filter when better than expert AND Q is underestimated
                    if q_overestimation is not None:
                        # q_overestimation < 0 means underestimation
                        q_underestimated = (q_overestimation < self.q_underestimation_threshold).float()  # (B, 1)
                        q_underestimated_expanded = q_underestimated.expand(-1, K)  # (B, K)
                        # Apply filter if BOTH conditions are true: better_than_expert AND q_underestimated
                        # Per-sample: should_filter_bk = better_than_expert_bk * q_underestimated_b
                        should_filter_per_sample = better_than_expert_per_sample * q_underestimated_expanded  # (B, K)
                        bc_filter_expanded = 1.0 - should_filter_per_sample  # (B, K)
                    else:
                        # Fallback to original behavior if q_overestimation not provided
                        bc_filter_expanded = 1.0 - better_than_expert_per_sample  # (B, K)
                else:
                    bc_filter_expanded = torch.ones_like(better_than_expert_per_sample, device=self.device)  # (B, K)
                
                # Override bc_filter for expert data if always_retain_bc_loss_for_expert_data is True
                if self.always_retain_bc_loss_for_expert_data and data_source is not None:
                    # For expert data (data_source == 1), always set bc_filter to 1.0
                    expert_mask = (data_source == 1).float()  # (B, 1) - 1.0 for expert, 0.0 for online
                    expert_mask_expanded = expert_mask.expand(-1, K)  # (B, K)
                    # Override bc_filter: 1.0 for expert data, keep original for online data
                    bc_filter_expanded = expert_mask_expanded + (1.0 - expert_mask_expanded) * bc_filter_expanded  # (B, K)
                
                # Keep a (B, 1) version for backward compatibility with logging
                bc_filter = bc_filter_expanded.mean(dim=1, keepdim=True)  # (B, 1) - average for logging
            
            # Q-value loss: -mean(Q(s, a_k)) (direct average, no softmax weights)
            q_loss_per_batch = q_current_samples.mean(dim=1)  # (B,) - average over K samples
            
            # Apply disable_q_loss_for_expert_data if enabled
            online_mask = None
            if self.disable_q_loss_for_expert_data and data_source is not None:
                # Mask Q loss for expert data: only apply Q loss to online data (data_source == 0)
                online_mask = (data_source == 0).float().squeeze(-1)  # (B, 1) -> (B,) - 1.0 for online, 0.0 for expert
                q_loss_per_batch = q_loss_per_batch * online_mask  # (B,)
            
            q_loss = -q_loss_per_batch.mean()  # scalar
            
            # Normalize Q-loss by mean absolute Q-value for stability (like FQL)
            if self.use_q_normalization:
                # Compute normalization constant only from the Q-values we're actually using for loss
                if online_mask is not None:
                    # Only normalize by online Q-values when we're masking expert data
                    online_mask_expanded = online_mask.unsqueeze(-1).expand(-1, K)  # (B, K)
                    online_q_values = q_current_samples * online_mask_expanded  # Zero out expert Q-values
                    online_count = online_mask_expanded.sum()  # Count of online samples
                    if online_count > 0:
                        q_abs_mean = (online_q_values.abs().sum() / online_count).detach()  # Mean of online Q-values only
                    else:
                        q_abs_mean = q_current_samples.abs().mean().detach()  # Fallback to all if no online samples
                else:
                    # Use all Q-values for normalization
                    q_abs_mean = q_current_samples.abs().mean().detach()
                
                if q_abs_mean > 1e-8:  # Avoid division by zero
                    q_loss_scale = 1.0 / q_abs_mean
                    q_loss = q_loss_scale * q_loss
            
            # Compute BC-style loss with filtering: ||current_action - pretrained_action||²
            action_diff = actions_samples - pretrained_actions_samples  # (B, K, horizon_steps, action_dim)
            # Compute MSE across action dimensions for each timestep, then average across time
            mse_per_timestep = (action_diff ** 2).mean(dim=-1)  # (B, K, horizon_steps)
            # Average across timesteps to get per-sample MSE: (B, K)
            mse_per_sample = mse_per_timestep.mean(dim=-1)  # (B, K)
            # Apply filtering and equal weighting (no softmax weights, just equal average)
            uniform_weights = torch.ones(B, K, device=self.device) / K  # (B, K) - equal weights
            weighted_filtered_mse = uniform_weights * bc_filter_expanded * mse_per_sample  # (B, K)
            # Sum across K samples and average across batch
            filtered_bc_loss = weighted_filtered_mse.sum(dim=1).mean()  # scalar
            
            total_loss = q_loss + self.bc_loss_weight * filtered_bc_loss
        
        # Compute additional metrics for logging
        with torch.no_grad():
            residual_norm = ((actions_samples - pretrained_actions_samples) ** 2).mean().sqrt()  # RMS of residual actions
            pretrained_q_values = avg_q_pretrained if not in_warmup else q_pretrained_samples.mean(dim=1, keepdim=True)
            q_values = avg_q_current if not in_warmup else q_current_samples.mean(dim=1, keepdim=True)
        
        return {
            'actor_total': total_loss,
            'actor_q_loss': q_loss,
            'actor_residual_loss': filtered_bc_loss,  # This is the filtered BC loss (equivalent to residual loss)
            'actor_bc_loss': filtered_bc_loss,  # BC-style loss for compatibility with logging
            # Metrics
            'q_advantage_mean': q_advantage.mean() if not in_warmup else torch.tensor(0.0, device=self.device),
            'better_than_expert_percentage': better_percentage,
            'pretrained_q_mean': pretrained_q_values.mean(),
            'current_q_mean': q_values.mean(),
            'residual_norm': residual_norm,
            'q_filtering_active': bc_filter.mean(),  # Mean of filtering mask
        }
    