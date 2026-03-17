"""
Distilled RL Model for online finetuning of flow matching policies.

This module implements the core components for online RL finetuning:
- DistilledActor: One-step distilled actor network
- Critic: Q-function network
- DistillResidualRLModel: Main model that orchestrates all components

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
from model.common.mlp import MLP

class DistilledActor(nn.Module):
    """
    One-step distilled actor network.
    
    Takes state s and noise z as input, outputs action a.
    This is a simple MLP that learns to map (s, z) -> a.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        cond_steps: int = 1,
        horizon_steps: int = 8,
        hidden_dims: List[int] = [256, 256, 256],
        activation_type: str = "Mish",
        use_layernorm: bool = False,
        **kwargs
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.cond_steps = cond_steps
        self.horizon_steps = horizon_steps
        
        # Input: flattened state + noise
        # state: (B, cond_steps, obs_dim) -> flattened to (B, cond_steps * obs_dim)
        # noise: (B, horizon_steps, action_dim) -> flattened to (B, horizon_steps * action_dim)
        input_dim = cond_steps * obs_dim + horizon_steps * action_dim
        output_dim = horizon_steps * action_dim
        
        mlp_dims = [input_dim] + hidden_dims + [output_dim]
        
        self.mlp = MLP(
            mlp_dims,
            activation_type=activation_type,
            out_activation_type="Identity",
            use_layernorm=use_layernorm,
        )
    
    def forward(self, state: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: (B, cond_steps, obs_dim) - chunked state
            noise: (B, horizon_steps, action_dim) - chunked noise
            
        Returns:
            action: (B, horizon_steps, action_dim) - predicted action chunk
        """
        B = noise.shape[0]  # batch size
        if isinstance(state, dict):
            state = state["state"]
        # Flatten inputs for processing
        state_flat = state.view(B, -1)  # (B, cond_steps * obs_dim)
        noise_flat = noise.view(B, -1)  # (B, horizon_steps * action_dim)
        
        # Concatenate flattened inputs
        x = torch.cat([state_flat, noise_flat], dim=-1)  # (B, cond_steps * obs_dim + horizon_steps * action_dim)
        action_flat = self.mlp(x)  # (B, horizon_steps * action_dim)
        
        # Reshape back to chunked format
        action = action_flat.view(B, self.horizon_steps, self.action_dim)  # (B, horizon_steps, action_dim)
        
        return action

class DistilledCritic(nn.Module):
    """
    Critic network that takes state, noise, and action as input.
    
    Q(s, z, a) -> scalar value or Q(s, a) -> scalar value
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        cond_steps: int = 1,
        horizon_steps: int = 8,
        hidden_dims: List[int] = [256, 256, 256],
        activation_type: str = "Mish",
        use_layernorm: bool = False,
        q_depends_on_noise: bool = False,  # If False, Q(s,a) instead of Q(s,z,a)
        critic_ensemble_size: int = 2,  # Number of Q-networks in ensemble
        conservative_q_method: str = "min",
        td_loss: str = "mse",  # TD loss type: "mse", "huber", "bce"
        **kwargs
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.cond_steps = cond_steps
        self.horizon_steps = horizon_steps
        self.q_depends_on_noise = q_depends_on_noise
        self.critic_ensemble_size = critic_ensemble_size
        self.conservative_q_method = conservative_q_method
        self.td_loss = td_loss
        
        # Input: flattened state + noise + action OR state + action
        # state: (B, cond_steps, obs_dim) -> flattened to (B, cond_steps * obs_dim)
        # noise: (B, horizon_steps, action_dim) -> flattened to (B, horizon_steps * action_dim) [optional]
        # action: (B, horizon_steps, action_dim) -> flattened to (B, horizon_steps * action_dim)
        if q_depends_on_noise:
            input_dim = cond_steps * obs_dim + 2 * horizon_steps * action_dim
        else:
            input_dim = cond_steps * obs_dim + horizon_steps * action_dim
        
        mlp_dims = [input_dim] + hidden_dims + [1]
        
        # Always use Identity activation - for BCE we'll use BCEWithLogitsLoss
        out_activation = "Identity"
        
        # Create ensemble of Q-networks
        self.Q_ensemble = nn.ModuleList([
            MLP(
                mlp_dims,
                activation_type=activation_type,
                out_activation_type=out_activation,
                use_layernorm=use_layernorm,
            )
            for _ in range(critic_ensemble_size)
        ])
    
    def forward(self, state: torch.Tensor, noise: torch.Tensor, action: torch.Tensor, return_all=False, return_mean=False) -> torch.Tensor:
        """
        Forward pass through ensemble of Q-networks.
        
        Args:
            state: (B, cond_steps, obs_dim) - chunked state
            noise: (B, horizon_steps, action_dim) - chunked noise (ignored if q_depends_on_noise=False)
            action: (B, horizon_steps, action_dim) - chunked action
            return_all: bool - if True, return all Q-values, else return conservative estimate
            
        Returns:
            If return_all=True: List of Q-values from all networks [(B, 1), ...]
            If return_all=False: Conservative Q-value estimate (B, 1) using specified method
        """
        return_mean = False # here i forced return min, i found using mean for actor loss comparable to min
        B = state.shape[0]  # batch size
        
        # Flatten all inputs for processing
        state_flat = state.view(B, -1)  # (B, cond_steps * obs_dim)
        action_flat = action.view(B, -1)  # (B, horizon_steps * action_dim)
        
        # Concatenate inputs based on whether Q depends on noise
        if self.q_depends_on_noise:
            noise_flat = noise.view(B, -1)  # (B, horizon_steps * action_dim)
            x = torch.cat([state_flat, noise_flat, action_flat], dim=-1)  # (B, cond_steps * obs_dim + 2 * horizon_steps * action_dim)
        else:
            x = torch.cat([state_flat, action_flat], dim=-1)  # (B, cond_steps * obs_dim + horizon_steps * action_dim)
        
        # Get Q-values from all networks in ensemble
        q_values = []
        for q_network in self.Q_ensemble:
            q_val = q_network(x)  # (B, 1) - raw logits if td_loss=="bce"
            
            # Apply sigmoid for BCE loss to get Q-values in [0,1] for everything except loss computation
            # (critic_loss method will use raw logits with BCEWithLogitsLoss)
            if self.td_loss == "bce" and not return_all:
                # Apply sigmoid when returning single Q-value for actor loss or evaluation
                q_val = torch.sigmoid(q_val)
                
            q_values.append(q_val)
        
        if return_all:
            # Return raw logits for critic_loss when td_loss=="bce"
            return q_values  # List of (B, 1) tensors
        else:
            # Stack Q-values for easier manipulation
            q_stacked = torch.stack(q_values, dim=0)  # (ensemble_size, B, 1)
            if return_mean:
                return q_stacked.mean(dim=0)  # (B, 1)
            if self.conservative_q_method == "min":
                # Return minimum Q-value for conservative estimate (this works)
                return torch.min(q_stacked, dim=0)[0]  # (B, 1)
            else:
                raise ValueError(f"Unknown conservative_q_method: {self.conservative_q_method}")
    
    def return_both(self, *args, **kwargs):
        """Backward compatibility method - now returns all Q-values"""
        return self.forward(*args, return_all=True, **kwargs)

class DistillResidualRLModel(nn.Module):
    """
    Main model for distilled RL finetuning.
    
    This orchestrates all components:
    - DistilledActor: π_θ(a|s,z)
    - DistilledCritic: Q_φ(s,z,a)
    - PretrainedFlowPolicy: π_pre(a|s,z) (frozen)
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        pretrained_flow_policy_path: str,
        # Network configurations
        actor_hidden_dims: List[int] = [256, 256, 256],
        critic_hidden_dims: List[int] = [256, 256, 256],
        activation_type: str = "Mish",
        # Loss coefficients
        bc_loss_weight: float = 1.0,
        critic_weight: float = 1.0,
        # Q-filtering parameters
        use_soft_q_filtering: bool = False,
        q_filtering_warmup_steps: int = 25000,
        q_underestimation_threshold: float = -0.1,  # Threshold for detecting Q underestimation
        # Exploration strategy warmup (separate from Q-filtering warmup)
        replay_flow_warmup_steps: int = 1000,
        # q normalization
        use_q_normalization: bool = False,
        # Q-function noise dependency
        q_depends_on_noise: bool = False,
        multi_sample_next_noise: bool = False,  # If True, use multiple samples for next noise
        num_next_noise_samples: int = 4,  # Number of samples for next noise (K)
        # Critic ensemble settings
        critic_ensemble_size: int = 2,  # Number of critic pairs in ensemble
        # Chunk parameters
        cond_steps: int = 1,
        horizon_steps: int = 4,
        # N-step returns
        use_n_step: bool = False,
        n_step: int = 1,
        # Disable Q loss for expert data (ablation)
        disable_q_loss_for_expert_data: bool = False,
        # Disable TD loss for expert data (ablation)
        disable_td_loss_for_expert_data: bool = False,
        always_retain_bc_loss_for_expert_data: bool = False,
        # TD loss type for critic
        td_loss: str = "mse",  # "mse", "huber", "bce"
        # Multi-z sampling for actor loss
        sample_multi_z_for_actor_loss: bool = False,
        num_multi_z_for_actor_loss: int = 8,
        condition_residual_on_base_action: bool = False,
        # Device
        device: str = "cuda",
        **kwargs
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.cond_steps = cond_steps
        self.horizon_steps = horizon_steps
        self.use_n_step = use_n_step
        self.n_step = n_step
        self.disable_q_loss_for_expert_data = disable_q_loss_for_expert_data
        self.disable_td_loss_for_expert_data = disable_td_loss_for_expert_data
        self.always_retain_bc_loss_for_expert_data = always_retain_bc_loss_for_expert_data
        self.td_loss = td_loss
        self.sample_multi_z_for_actor_loss = sample_multi_z_for_actor_loss
        self.num_multi_z_for_actor_loss = num_multi_z_for_actor_loss
        self.device = device
        
        # Loss coefficients
        self.bc_loss_weight = bc_loss_weight
        self.critic_weight = critic_weight
        
        # Q-filtering settings
        self.use_soft_q_filtering = use_soft_q_filtering
        self.q_filtering_warmup_steps = q_filtering_warmup_steps
        self.replay_flow_warmup_steps = replay_flow_warmup_steps
        self.q_underestimation_threshold = q_underestimation_threshold
        self.use_q_normalization = use_q_normalization
        self.q_depends_on_noise = q_depends_on_noise
        self.multi_sample_next_noise = multi_sample_next_noise
        self.num_next_noise_samples = num_next_noise_samples
        self.critic_ensemble_size = critic_ensemble_size
        self.conservative_q_method = kwargs.get('conservative_q_method', 'min')
        print(f"Critic ensemble size: {self.critic_ensemble_size}")
        print(f"Conservative Q method: {self.conservative_q_method}")

        self.condition_residual_on_base_action = condition_residual_on_base_action
        # Initialize networks with explicit dimensions for chunked data
        use_layernorm_val = kwargs.pop('use_layernorm', True)

        self.actor = DistilledActor(
            obs_dim=obs_dim,
            action_dim=action_dim,
            cond_steps=cond_steps,
            horizon_steps=horizon_steps,
            hidden_dims=actor_hidden_dims,
            activation_type=activation_type,
            use_layernorm=use_layernorm_val,
            **kwargs
        ).to(device)
        
        self.critic = DistilledCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            cond_steps=cond_steps,
            horizon_steps=horizon_steps,
            hidden_dims=critic_hidden_dims,
            activation_type=activation_type,
            q_depends_on_noise=q_depends_on_noise,
            critic_ensemble_size=critic_ensemble_size,
            conservative_q_method=kwargs.get('conservative_q_method', 'min'),
            td_loss=td_loss,
            use_layernorm=use_layernorm_val
        ).to(device)
        
        # Target critic for stable Q-learning (SAC-style)
        self.target_critic = DistilledCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            cond_steps=cond_steps,
            horizon_steps=horizon_steps,
            hidden_dims=critic_hidden_dims,
            activation_type=activation_type,
            q_depends_on_noise=q_depends_on_noise,
            critic_ensemble_size=critic_ensemble_size,
            conservative_q_method=kwargs.get('conservative_q_method', 'min'),
            td_loss=td_loss,
            use_layernorm=use_layernorm_val
        ).to(device)
        
        # Initialize target critic with same weights as critic
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # Load pretrained flow matching policy using hydra.instantiate pattern
        self.pretrained_flow_policy = self._load_pretrained_policy(
            pretrained_flow_policy_path, device
        )

        log.info(f"DistillResidualRLModel initialized with:")
        log.info(f"  obs_dim: {obs_dim}")
        log.info(f"  action_dim: {action_dim}")
        log.info(f"  td_loss: {td_loss}")

    def _load_pretrained_policy(self, checkpoint_path: str, device: str):
        """
        Load pretrained flow matching policy using hydra.instantiate pattern.
        
        This follows the same pattern as eval_agent.py:
        1. Load the saved hydra config from the checkpoint directory
        2. Use hydra.instantiate to create the model
        3. Load the state dict from the checkpoint
        
        Args:
            checkpoint_path: Path to the pretrained policy checkpoint (.pt file)
            device: Device to load the model on
            
        Returns:
            Loaded and frozen FlowMatchingModel
        """
        # Extract the directory containing the checkpoint
        checkpoint_dir = os.path.dirname(checkpoint_path)
        
        # Find the .hydra directory (go up until we find it)
        current_dir = checkpoint_dir
        hydra_config_path = None
        for _ in range(5):  # Limit search depth
            hydra_dir = os.path.join(current_dir, '.hydra')
            if os.path.exists(hydra_dir):
                hydra_config_path = os.path.join(hydra_dir, 'config.yaml')
                break
            current_dir = os.path.dirname(current_dir)
        
        if hydra_config_path is None or not os.path.exists(hydra_config_path):
            raise FileNotFoundError(f"Could not find .hydra/config.yaml for checkpoint {checkpoint_path}")
        
        # Load the hydra config and manually resolve eval interpolations
        log.info(f"Loading pretrained policy config from: {hydra_config_path}")
        
        # Read the config file as text first
        with open(hydra_config_path, 'r') as f:
            config_text = f.read()
        
        # Simple fix for the specific eval expression we know exists
        # Replace ${eval:'${obs_dim} * ${cond_steps}'} with a computed value
        if "${eval:'${obs_dim} * ${cond_steps}'}" in config_text:
            # Get obs_dim and cond_steps from a simple YAML load
            import yaml
            config_dict = yaml.safe_load(config_text)
            obs_dim = config_dict['obs_dim']
            cond_steps = config_dict['cond_steps']
            computed_cond_dim = obs_dim * cond_steps
            
            config_text = config_text.replace("${eval:'${obs_dim} * ${cond_steps}'}", str(computed_cond_dim))
            log.info(f"Resolved eval interpolation: obs_dim * cond_steps = {obs_dim} * {cond_steps} = {computed_cond_dim}")
        
        # Now load the processed config with OmegaConf
        pretrained_cfg = OmegaConf.create(config_text)
        
        # Instantiate the model using the processed config
        pretrained_model = hydra.utils.instantiate(pretrained_cfg.model)
        pretrained_model.to(device)
        
        # Load the checkpoint
        log.info(f"Loading pretrained policy weights from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        
        # Load the model state dict (use 'model' key like eval_agent.py)
        if 'model' in ckpt:
            pretrained_model.load_state_dict(ckpt['model'])
        else:
            # Fallback: try to load state dict directly
            pretrained_model.load_state_dict(ckpt)
        
        # Special handling for DiffusionModel: Enable DDIM for deterministic sampling in RL
        if pretrained_model.__class__.__name__ == 'DiffusionModel':
            log.info("Detected DiffusionModel - configuring for DDIM deterministic sampling")
            
            # Set DDIM parameters for deterministic sampling
            pretrained_model.use_ddim = True
            
            # Use a reasonable number of DDIM steps if not already set
            if not hasattr(pretrained_model, 'ddim_steps') or pretrained_model.ddim_steps is None:
                pretrained_model.ddim_steps = 10  # Default to 10 DDIM steps
                log.info(f"Setting DDIM steps to {pretrained_model.ddim_steps}")
            
            # Re-initialize DDIM parameters since use_ddim was changed
            # This mirrors the initialization in DiffusionModel.__init__
            device = pretrained_model.betas.device
            denoising_steps = pretrained_model.denoising_steps
            ddim_steps = pretrained_model.ddim_steps
            
            # DDIM sampling parameters (from diffusion.py lines 155-196)
            step_ratio = denoising_steps // ddim_steps
            pretrained_model.ddim_t = (
                torch.arange(0, ddim_steps, device=device) * step_ratio
            )
            
            pretrained_model.ddim_alphas = (
                pretrained_model.alphas_cumprod[pretrained_model.ddim_t].clone().to(torch.float32)
            )
            pretrained_model.ddim_alphas_sqrt = torch.sqrt(pretrained_model.ddim_alphas)
            pretrained_model.ddim_alphas_prev = torch.cat(
                [
                    torch.tensor([1.0]).to(torch.float32).to(device),  # IMPORTANT: Must be 1.0, not alphas_cumprod[0]
                    pretrained_model.alphas_cumprod[pretrained_model.ddim_t[:-1]],
                ]
            )
            pretrained_model.ddim_sqrt_one_minus_alphas = (1.0 - pretrained_model.ddim_alphas) ** 0.5
            
            # DDIM eta = 0 for deterministic sampling
            ddim_eta = 0
            pretrained_model.ddim_sigmas = (
                ddim_eta
                * (
                    (1 - pretrained_model.ddim_alphas_prev)
                    / (1 - pretrained_model.ddim_alphas)
                    * (1 - pretrained_model.ddim_alphas / pretrained_model.ddim_alphas_prev)
                )
                ** 0.5
            )
            
            # Flip all DDIM tensors for reverse process
            pretrained_model.ddim_t = torch.flip(pretrained_model.ddim_t, [0])
            pretrained_model.ddim_alphas = torch.flip(pretrained_model.ddim_alphas, [0])
            pretrained_model.ddim_alphas_sqrt = torch.flip(pretrained_model.ddim_alphas_sqrt, [0])
            pretrained_model.ddim_alphas_prev = torch.flip(pretrained_model.ddim_alphas_prev, [0])
            pretrained_model.ddim_sqrt_one_minus_alphas = torch.flip(
                pretrained_model.ddim_sqrt_one_minus_alphas, [0]
            )
            pretrained_model.ddim_sigmas = torch.flip(pretrained_model.ddim_sigmas, [0])
            
            log.info(f"DDIM configured with {ddim_steps} steps for deterministic sampling")
        
        # Move model to device again after loading state dict to ensure all components are on correct device
        pretrained_model.to(device)
        
        # Ensure all buffers are also moved to device
        for buffer in pretrained_model.buffers():
            buffer.data = buffer.data.to(device)
        
        # Freeze the pretrained model
        for param in pretrained_model.parameters():
            param.requires_grad = False
        pretrained_model.eval()
        
        model_type = "DiffusionModel" if pretrained_model.__class__.__name__ == 'DiffusionModel' else "FlowMatchingModel"
        log.info(f"Pretrained {model_type} loaded and frozen successfully")
        return pretrained_model
    
    def update_target_networks(self, tau: float = 0.005):
        """
        Update target networks using Polyak averaging.
        
        Args:
            tau: Polyak averaging coefficient (target = tau * current + (1-tau) * target)
        """
        with torch.no_grad():
            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
    

    def get_exploration_action(self, state: torch.Tensor, num_samples: int = 10, 
                              exploration_strategy: str = "max_q_std", training_step: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get exploration action using specified strategy.
        
        This method is used for online exploration to maximize diversity in collected data.
        Supports two strategies:
        - max_q_std: Select action with highest Q-std across ensemble (epistemic uncertainty)
        - max_q_min: Select action with highest minimum Q-value across ensemble (optimistic)
        
        If training_step < replay_flow_warmup_steps, uses single sample regardless of strategy.
        
        Args:
            state: (B, cond_steps, obs_dim) - current state
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

            elif exploration_strategy == 'max_q_std':  # max_q_std (default)
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
            else:
                raise ValueError(f"Unknown exploration strategy: {exploration_strategy}")
            # Reshape actions back
            actions_reshaped = actions_flat.view(num_samples, B, self.horizon_steps, self.action_dim)
            
            # Select actions based on strategy
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
            # WARMUP PHASE: Simple loss without filtering or self-imitation
            
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
    
    def critic_loss(
        self,
        state: torch.Tensor,
        noise: torch.Tensor,
        action: torch.Tensor,
        target_q: torch.Tensor,
        data_source: Optional[torch.Tensor] = None,  # (B,1) - 0 for online, 1 for expert
    ) -> Dict[str, torch.Tensor]:
        """
        Compute critic loss for both Q-networks (double Q-learning).
        
        Args:
            state: (B, cond_steps, obs_dim) - chunked current state
            noise: (B, horizon_steps, action_dim) - chunked noise
            action: (B, horizon_steps, action_dim) - chunked action
            target_q: (B, 1) - target Q-values
            data_source: (B, 1) - 0 for online, 1 for expert data (optional)
            
        Returns:
            loss_dict: Dictionary containing loss components
        """
        # Get predictions from all Q-networks in ensemble
        q_all = self.critic(state, noise, action, return_all=True)
        
        # Compute loss for all networks in ensemble
        total_loss = 0
        loss_dict = {}
        
        for i, q_pred in enumerate(q_all):
            # Compute TD loss based on selected loss type
            if self.td_loss == "mse":
                # Mean Squared Error loss
                if self.disable_td_loss_for_expert_data and data_source is not None:
                    # Mask TD loss for expert data: only apply TD loss to online data (data_source == 0)
                    online_mask = (data_source == 0).float()  # (B, 1) - 1.0 for online, 0.0 for expert
                    # Compute per-sample MSE loss
                    per_sample_loss = F.mse_loss(q_pred, target_q, reduction='none')  # (B, 1)
                    # Apply mask and reduce
                    q_loss = (per_sample_loss * online_mask).mean()
                else:
                    # Standard TD loss for all samples
                    q_loss = F.mse_loss(q_pred, target_q)
                    
            elif self.td_loss == "huber":
                # Huber loss (smooth L1 loss) - more robust to outliers
                if self.disable_td_loss_for_expert_data and data_source is not None:
                    online_mask = (data_source == 0).float()
                    per_sample_loss = F.smooth_l1_loss(q_pred, target_q, reduction='none', beta=1.0)  # (B, 1)
                    q_loss = (per_sample_loss * online_mask).mean()
                else:
                    q_loss = F.smooth_l1_loss(q_pred, target_q, beta=1.0)
                    
            elif self.td_loss == "bce":
                # Binary Cross Entropy with Logits loss
                # q_pred are raw logits, target_q should be in [0,1]
                # Assert that target values are in valid range
                assert torch.all(target_q >= 0.0) and torch.all(target_q <= 1.0), \
                    f"BCE loss requires target Q-values in [0,1], got range [{target_q.min():.4f}, {target_q.max():.4f}]"
                
                if self.disable_td_loss_for_expert_data and data_source is not None:
                    online_mask = (data_source == 0).float()
                    # Use BCEWithLogitsLoss for numerical stability
                    per_sample_loss = F.binary_cross_entropy_with_logits(q_pred, target_q, reduction='none')  # (B, 1)
                    q_loss = (per_sample_loss * online_mask).mean()
                else:
                    q_loss = F.binary_cross_entropy_with_logits(q_pred, target_q)
            else:
                raise ValueError(f"Unknown td_loss type: {self.td_loss}")
            
            total_loss += q_loss
            
            # Store individual losses for debugging (first 3 critics)
            if i < 3:
                loss_dict[f'q{i+1}_loss'] = q_loss
        
        loss_dict['critic_loss'] = total_loss
        return loss_dict
    
    def get_action(self, state: torch.Tensor, noise: torch.Tensor, return_pretrained_actions: bool = False) -> torch.Tensor:
        """
        Get action as sum of pretrained policy and residual actor.
        
        Action = π_pre(s,z) + r_θ(s,z) or π_pre(s,z) + r_θ(s,a_base)
        
        Args:
            state: (B, cond_steps, obs_dim) - current state
            noise: (B, horizon_steps, action_dim) - noise for action generation
            return_pretrained_actions: If True, return tuple (total_actions, pretrained_actions)
            
        Returns:
            action: (B, horizon_steps, action_dim) - total action (pretrained + residual)
            OR if return_pretrained_actions:
            (action, pretrained_actions): tuple of actions
        """
        # Get pretrained action (no gradient)
        with torch.no_grad():
            cond = {"state": state}
            pretrained_sample = self.pretrained_flow_policy(cond, deterministic=False, init_noise=noise)
            pretrained_actions = pretrained_sample.trajectories  # (B, horizon_steps, action_dim)
        
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
        return total_actions
    
    def loss(
        self,
        state: torch.Tensor,
        noise: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        gamma: float = 0.99,
        training_step: int = 0,
        q_overestimation: Optional[torch.Tensor] = None,
        n_steps: Optional[torch.Tensor] = None,
        data_source: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all losses for the distilled RL model.
        
        Args:
            state: (B, cond_steps, obs_dim) - chunked current state
            noise: (B, horizon_steps, action_dim) - initial noise for action generation
            action: (B, horizon_steps, action_dim) - action chunk
            next_state: (B, cond_steps, obs_dim) - chunked next state
            reward: (B, 1) - reward
            done: (B, 1) - done flag
            gamma: float - discount factor
            
        Returns:
            loss_dict: Dictionary containing all loss components
        """
        batch_size = state.shape[0]
        
        # All models work with the entire chunks - no need to extract single states/actions
        # Get current actions from actor for Q-value computation
        current_actions = self.get_action(state, noise)

        q_values = self.critic(state, noise, current_actions)  # (B, 1)

        # Compute target Q-values using target networks (SAC-style)
        with torch.no_grad():
            # Sample noise for next actions (always single sample for actor loss consistency)
            next_noise = torch.randn(batch_size, action.shape[1], self.action_dim, device=self.device)
            
            if not self.multi_sample_next_noise:
                next_actions = self.get_action(next_state, next_noise)  # (B, horizon_steps, action_dim) - use get_action for residual RL compatibility
                # Use target critic ensemble for stable targets
                target_next_q = self.target_critic(next_state, next_noise, next_actions)  # (B, 1) - already min across ensemble
                
                # Apply appropriate discount factor based on training configuration
                if self.use_n_step:
                    # For n-step returns, use gamma^n_step
                    gamma_effective = gamma ** n_steps.float()  # (B, 1)
                else:
                    # Default single-step return
                    gamma_effective = gamma
                    
                target_q = reward + gamma_effective * (1 - done.float()) * target_next_q  # (B, 1) - reward already includes intrinsic or n-step rewards
            else:
                # Multi-sample for more stable Q-targets
                K = self.num_next_noise_samples
                next_noise_samples = torch.randn(K, batch_size, action.shape[1], self.action_dim, device=self.device)
                next_state_rep = next_state.unsqueeze(0).expand(K, -1, -1, -1).reshape(K*batch_size, *next_state.shape[1:])
                next_noise_flat = next_noise_samples.reshape(K*batch_size, *next_noise_samples.shape[2:])
                next_actions = self.get_action(next_state_rep, next_noise_flat)  # (K*B, horizon_steps, action_dim)

                target_q_samples = self.target_critic(next_state_rep, next_noise_flat, next_actions)  # (K*B, 1) - already min across ensemble
                
                # Standard mean aggregation
                target_next_q = target_q_samples.reshape(K, batch_size, 1).mean(dim=0)  # (B, 1)
                
                # Apply appropriate discount factor based on training configuration
                if self.use_n_step:
                    # For n-step returns, use gamma^n_step
                    gamma_effective = gamma ** n_steps.float()  # (B, 1)
                else:
                    # Default single-step return
                    gamma_effective = gamma
                    
                target_q = reward + gamma_effective * (1 - done.float()) * target_next_q  # (B, 1) - reward already includes intrinsic or n-step rewards
       
        # Compute all losses (pass pre-computed current_actions to avoid redundant call)
        actor_losses = self.actor_loss(
            state, noise, current_actions, q_values, next_state, next_noise, training_step, q_overestimation, data_source=data_source
        )
        critic_losses = self.critic_loss(state, noise, action, target_q)  # Use dataset actions for critic

        # Combine all losses
        total_loss = (
            actor_losses['actor_total'] +
            self.critic_weight * critic_losses['critic_loss']
            )
        
        # Return all losses
        return {
            'total_loss': total_loss,
            **actor_losses,
            **critic_losses,
        }
    
