"""
Image-based finetune agent for distilled flow matching policies.

This agent extends TrainDistillFlowAgent to handle image observations by:
1. Using the visual feature extraction and merging in the environment wrapper
2. Using standard ReplayBuffer with augmented state dimension
3. Pre-processing expert dataset with merged visual features + state
4. Maximizing inheritance from parent class
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import logging
import hydra
import wandb
from typing import Dict, Optional
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

log = logging.getLogger(__name__)
from agent.finetune.train_distill_residual_flow_agent import TrainDistillResidualFlowAgent
from util.hybrid_replay_buffer import HybridReplayBuffer

class TrainDistillResidualFlowImgAgent(TrainDistillResidualFlowAgent):
    """
    Image-based distilled flow matching agent.

    Extends TrainDistillResidualFlowAgent to handle image observations with visual features
    merged into the state by the environment wrapper.
    """
    
    def __init__(self, cfg):
        # Store original obs_dim before augmentation
        self.original_obs_dim = cfg.obs_dim
        
        # Feature caching to avoid redundant extraction
        self._feature_cache = {}
        
        # Extract visual feature dimensions from pretrained policy's config
        pretrained_config_path = os.path.join(
            os.path.dirname(cfg.base_policy_path), 
            "..", 
            ".hydra", 
            "config.yaml"
        )
        pretrained_config_path = os.path.normpath(pretrained_config_path)
        
        log.info(f"Loading pretrained config from: {pretrained_config_path}")
        
        pretrained_config = OmegaConf.load(pretrained_config_path)
        
        # Verify dataset configuration matches
        pretrained_dataset_path = pretrained_config.train_dataset_path
        current_dataset_path = cfg.expert_dataset.dataset_path
        
        # Normalize paths for comparison (resolve any env variables)
        pretrained_dataset_path = os.path.expandvars(pretrained_dataset_path)
        current_dataset_path = os.path.expandvars(current_dataset_path)
        
        # Verify max_n_episodes matches if available
        pretrained_max_episodes = pretrained_config.train_dataset.max_n_episodes
        current_max_episodes = cfg.expert_dataset.max_n_episodes
        
        assert pretrained_max_episodes == current_max_episodes, (
            f"Max episodes mismatch!\n"
            f"  Pretrained model used: {pretrained_max_episodes} episodes\n"
            f"  Current config uses: {current_max_episodes} episodes\n"
            f"  Please ensure max_n_episodes matches for consistent training."
        )
        log.info(f"✓ Max episodes verified: {current_max_episodes}")
        
        # Verify flow_steps/ddim_steps matches based on model type
        if hasattr(pretrained_config, 'flow_steps'):
            # Flow matching model
            pretrained_flow_steps = pretrained_config.flow_steps
            current_flow_steps = cfg.flow_steps
            
            assert pretrained_flow_steps == current_flow_steps, (
                f"Flow steps mismatch!\n"
                f"  Pretrained model used: {pretrained_flow_steps} flow steps\n"
                f"  Current config uses: {current_flow_steps} flow steps\n"
                f"  Please ensure flow_steps matches the pretrained model for consistency."
            )
            log.info(f"✓ Flow steps verified: {current_flow_steps}")
        elif hasattr(pretrained_config, 'denoising_steps'):
            # Diffusion model - check denoising steps
            pretrained_denoising_steps = pretrained_config.denoising_steps
            
            # For diffusion models, we use DDIM at inference time
            # The current config should specify ddim_steps for inference
            if hasattr(cfg, 'ddim_steps'):
                current_ddim_steps = cfg.ddim_steps
                log.info(f"✓ Diffusion model with {pretrained_denoising_steps} denoising steps")
                log.info(f"  Will use {current_ddim_steps} DDIM steps for deterministic inference")
            else:
                log.warning("No ddim_steps specified in current config for diffusion model")
        else:
            # Fallback if neither is found
            log.warning("Could not verify flow/DDIM steps - neither flow_steps nor ddim_steps found in pretrained config")

        # Check network type for visual feature dimension extraction
        network_target = pretrained_config.model.network._target_
        
        self.visual_feature_dim = pretrained_config.model.network.visual_feature_dim
        model_type = "VisionResNetDiffusionMLP" if "Diffusion" in network_target else "VisionResNetFlowMatchingMLP"
        log.info(f"Using {model_type} visual_feature_dim: {self.visual_feature_dim}")

        # Update obs_dim to include visual features BEFORE calling parent init
        self.obs_dim = self.visual_feature_dim + self.original_obs_dim
        cfg.obs_dim = self.obs_dim  # Update config for model creation

        log.info(f"TrainDistillResidualFlowImgAgent config update:")
        log.info(f"  Original obs_dim: {self.original_obs_dim}")
        log.info(f"  Visual feature dim: {self.visual_feature_dim}")
        log.info(f"  Updated obs_dim: {self.obs_dim}")
        
        # Call parent init with updated obs_dim
        # Note: Wrapper returns raw images since we can't pass model to async processes
        super().__init__(cfg)
        print('deug obs_dim after super init:', self.obs_dim, cfg.obs_dim)
        # Store pretrained model in eval mode for feature extraction
        self.pretrained_model = self.model.pretrained_flow_policy
        self.pretrained_model.eval()
        
        # Exploration strategy settings
        self.online_explore_strategy = cfg.get("online_explore_strategy", "standard")
        self.evaluate_strategy = cfg.get("evaluate_strategy", "standard")
        self.num_exploration_samples = cfg.get("num_exploration_samples", 10)
        
        self.current_training_step = 0

        log.info(f"TrainDistillResidualFlowImgAgent initialized successfully")
        log.info(f"  Model obs_dim: {self.model.obs_dim}")
        log.info(f"  Visual feature extraction will be done in agent")

        if self.online_explore_strategy != "standard":
            log.info(f"  Online exploration strategy: {self.online_explore_strategy} with {self.num_exploration_samples} samples")
        if self.evaluate_strategy != "standard":
            log.info(f"  Evaluation strategy: {self.evaluate_strategy}")
    
    def _setup_replay_buffer(self, cfg):
        """
        Override parent's replay buffer setup to handle expert dataset preprocessing.
        Uses standard ReplayBuffer but with augmented obs_dim.
        """
        # Pre-process expert dataset if RLPD is enabled
        expert_dataset_with_features = None
        if cfg.get("use_rlpd", False) and hasattr(cfg, 'expert_dataset'):
            # Pre-process expert dataset to include merged visual features + state
            log.info("Pre-processing expert dataset with visual features...")
            expert_dataset_with_features = self._preprocess_expert_dataset(
                hydra.utils.instantiate(cfg.expert_dataset)
            )
            log.info(f"Expert dataset pre-processed: {len(expert_dataset_with_features)} transitions")
        
        # Create replay buffer with augmented obs_dim
        self.replay_buffer = HybridReplayBuffer(
            obs_dim=self.obs_dim,  # This is now augmented (visual_feature_dim + original_obs_dim)
            action_dim=self.action_dim,
            noise_dim=self.noise_dim,
            max_size=cfg.replay_buffer.get("max_size", 100000),
            n_envs=self.n_envs,
            cond_steps=self.cond_steps,
            horizon_steps=self.horizon_steps,
            device=self.device,
            gamma=cfg.train.get("gamma", 0.99),
            log_q_overestimation=cfg.get("log_q_overestimation", False),
            # RLPD settings
            use_rlpd=cfg.get("use_rlpd", False),
            expert_ratio=cfg.get("expert_ratio", 0.5),
            expert_dataset=expert_dataset_with_features,
            # N-step returns
            use_n_step=cfg.replay_buffer.get("use_n_step", False),
            n_step=cfg.replay_buffer.get("n_step", 1),
            # Expert dataset n-step settings (can differ from online data)
            expert_use_n_step=cfg.replay_buffer.get("expert_use_n_step", False),
            expert_n_step=cfg.replay_buffer.get("expert_n_step", 1),
        )
    
    def _preprocess_expert_dataset(self, expert_dataset):
        """
        Pre-process expert dataset to merge visual features with state.
        
        This extracts visual features from RGB observations using the pretrained visual encoder
        and merges them with state observations to match the augmented state format.
        
        Args:
            expert_dataset: StitchedSequenceQLearningDataset with image data
            
        Returns:
            List of transitions with merged state (visual features + low-dim state)
        """
        log.info("Extracting and merging visual features for expert dataset...")
        
        # Get pretrained model for feature extraction
        pretrained_model = self.model.pretrained_flow_policy
        pretrained_model.eval()
        
        processed_transitions = []
        
        # Process in batches to save memory
        batch_size = 32
        for i in range(0, len(expert_dataset), batch_size):
            batch_end = min(i + batch_size, len(expert_dataset))
            batch_transitions = [expert_dataset[j] for j in range(i, batch_end)]
            
            # Extract conditions (states and images) from batch
            states_batch = []
            rgb_batch = []
            next_states_batch = []
            next_rgb_batch = []
            
            for transition in batch_transitions:
                # transition.conditions is a dict with 'state', 'rgb', 'next_state', and 'next_rgb' keys
                condition = transition.conditions
                states_batch.append(condition['state'])
                rgb_batch.append(condition['rgb'])
                next_states_batch.append(condition['next_state'])
                next_rgb_batch.append(condition['next_rgb'])
            
            # Convert to tensors
            states_tensor = torch.stack([torch.from_numpy(s) if isinstance(s, np.ndarray) else s 
                                         for s in states_batch]).float().to(self.device)
            rgb_tensor = torch.stack([torch.from_numpy(r) if isinstance(r, np.ndarray) else r 
                                     for r in rgb_batch]).float().to(self.device)
            next_states_tensor = torch.stack([torch.from_numpy(s) if isinstance(s, np.ndarray) else s 
                                             for s in next_states_batch]).float().to(self.device)
            next_rgb_tensor = torch.stack([torch.from_numpy(r) if isinstance(r, np.ndarray) else r 
                                          for r in next_rgb_batch]).float().to(self.device)
            
            # Extract visual features using pretrained model
            with torch.no_grad():
                # Prepare condition dict for feature extraction
                cond = {
                    'state': states_tensor,  # Already has cond_steps dimension from dataset
                    'rgb': rgb_tensor  # Already has cond_steps dimension from dataset
                }
                # Extract features (these already contain both visual and state features)
                merged_features = pretrained_model.network.extract_visual_features(cond)  # (batch, feature_dim)
                
                # For next state, use proper next_rgb
                next_cond = {
                    'state': next_states_tensor,
                    'rgb': next_rgb_tensor  # Using proper next RGB observations
                }
                next_merged_features = pretrained_model.network.extract_visual_features(next_cond)
            
            # Create processed transitions with merged features
            for j, transition in enumerate(batch_transitions):
                # Get original state (for concatenation with visual features)
                original_state = transition.conditions['state']  # (cond_steps, obs_dim)
                original_next_state = transition.conditions.get('next_state', original_state)
                
                # Merge visual features with original state
                # The merged_features already contains processed visual+state, but we need to 
                # concatenate with original state to match wrapper's output format
                visual_feat = merged_features[j] # (feature_dim,)
                next_visual_feat = next_merged_features[j]
                
                augmented_state = torch.cat([
                    original_state,
                    visual_feat.unsqueeze(0).repeat(self.cond_steps, 1)
                ], dim=-1)  # (cond_steps, augmented_obs_dim)
                
                augmented_next_state = torch.cat([
                    original_next_state,
                    next_visual_feat.unsqueeze(0).repeat(self.cond_steps, 1)
                ], dim=-1)

                # Create new transition with augmented states
                from collections import namedtuple
                # Use same Transition structure as parent expects
                from agent.dataset.sequence import Transition
                
                processed_transition = Transition(
                    actions=transition.actions,
                    conditions={
                        'state': augmented_state,
                        'next_state': augmented_next_state
                    },
                    rewards=transition.rewards,
                    dones=transition.dones,
                    mc_return=transition.mc_return if hasattr(transition, 'mc_return') else transition.rewards
                )
                
                processed_transitions.append(processed_transition)
            
            if (i + batch_size) % (batch_size * 100) == 0:
                log.info(f"Processed {min(i + batch_size, len(expert_dataset))}/{len(expert_dataset)} expert transitions")
        
        log.info(f"Expert dataset pre-processing complete: {len(processed_transitions)} transitions")
        return processed_transitions
    
    def _extract_visual_features_from_obs(self, obs_venv):
        """
        Extract visual features from raw observations using pretrained model.
        
        Args:
            obs_venv: Dict with 'rgb' and 'state' keys from environment
            
        Returns:
            Augmented state with visual features merged
        """
        # Get raw state and rgb from obs
        state = obs_venv["state"]  # (n_envs, cond_steps, obs_dim) or (n_envs, obs_dim)
        rgb = obs_venv["rgb"]  # (n_envs, cond_steps, H, W, C) or (n_envs, H, W, C)
        
        # Convert to tensors
        if not isinstance(state, torch.Tensor):
            state = torch.from_numpy(state).float().to(self.device)
        if not isinstance(rgb, torch.Tensor):
            rgb = torch.from_numpy(rgb).float().to(self.device)
        
        # Ensure proper dimensions
        if state.ndim == 2:
            state = state.unsqueeze(1)  # Add cond_steps dim
        if rgb.ndim == 4:
            rgb = rgb.unsqueeze(1)  # Add cond_steps dim
        
        # Handle RGB format conversion (H, W, C) -> (C, H, W)
        if rgb.shape[-1] % 3 == 0:  # Channel last format
            rgb = rgb.permute(0, 1, 4, 2, 3)  # (n_envs, cond_steps, C, H, W)
        
        # Prepare condition dict for feature extraction
        cond = {
            'state': state,
            'rgb': rgb
        }
        
        # Extract visual features using pretrained model
        with torch.no_grad():
            visual_features = self.pretrained_model.network.extract_visual_features(cond)
            # visual_features shape: (n_envs, feature_dim)
        
        # Merge visual features with original state
        # Repeat visual features for each cond_step
        if state.shape[1] > 1:  # Has cond_steps dimension
            visual_features = visual_features.unsqueeze(1).repeat(1, state.shape[1], 1)
        else:
            visual_features = visual_features.unsqueeze(1)
        
        augmented_state = torch.cat([state, visual_features], dim=-1)
        
        return augmented_state  # Return as tensor
    
    def get_action(self, obs_venv, strategy="standard"):
        """
        Override to extract visual features before getting action.
        
        Args:
            obs_venv: Dict with 'rgb' and 'state' keys
            strategy: str - name of the strategy to use (e.g., "standard", "max_q_min", "max_q_std")
            
        Returns:
            action_venv: Actions for each environment
            noise_venv: Noise used for each environment
        """
        with torch.no_grad():
            # Extract visual features and create augmented state tensor
            augmented_state = self._extract_visual_features_from_obs(obs_venv)  # Returns tensor
            
            # Generate actions based on the specified strategy
            if strategy != "standard":
                action_venv, noise_venv = self.model.get_exploration_action(
                    augmented_state, 
                    num_samples=self.num_exploration_samples,
                    exploration_strategy=strategy,
                    training_step=self.current_training_step  # Use current training step from parent's run()
                )
            else:
                # Standard action generation (single sample with random noise)
                # Use batch size from augmented_state to handle both training (n_envs) and evaluation (eval_n_envs)
                batch_size = augmented_state.shape[0]
                noise_venv = torch.randn(batch_size, self.horizon_steps, self.action_dim, device=self.device)
                action_venv = self.model.get_action(augmented_state, noise_venv)

        return action_venv, noise_venv
    
    def collect_transition(self, obs_venv, action_venv, noise_venv, reward_venv, done_venv, next_obs_venv, info_venv=None):
        """        
        All inputs are already tensors from the parent's run() method.
        """
        full_trajectory_info = []

        for env_idx in range(self.n_envs):
            traj = info_venv[env_idx]['full_trajectory']
            
            # Copy trajectory structure
            processed_traj = {
                'initial_obs': None,
                'observations': [],
                'actions': traj['actions'],
                'rewards': traj['rewards'], 
                'dones': traj['dones'],
                'include_initial': traj['include_initial'],
            }
            
            # Process initial observation if present
            if traj['initial_obs'] is not None:
                # For image envs, initial_obs is a dict with 'state' and 'rgb' keys
                # Add batch dimension for _extract_visual_features_from_obs
                init_obs_batch = {k: np.array([v]) for k, v in traj['initial_obs'].items()}
                
                # Extract visual features - returns tensor with batch dim
                augmented_init = self._extract_visual_features_from_obs(init_obs_batch)
                # Remove batch dimension but keep as tensor
                processed_traj['initial_obs'] = augmented_init[0]
            
            # Batch process all step observations
            # For image envs, observations is a list of dicts with 'state' and 'rgb' keys
        
            # Stack all observations into a batch - each observation is a dict
            obs_batch = {}
            for key in traj['observations'][0].keys():  # Get keys from first observation
                obs_batch[key] = np.array([obs[key] for obs in traj['observations']])
            
            # Extract visual features for all observations at once
            augmented_obs = self._extract_visual_features_from_obs(obs_batch)
            
            # Split back into list of individual observations (keep as tensors)
            for i in range(len(traj['observations'])):
                processed_traj['observations'].append(augmented_obs[i])
            
            full_trajectory_info.append(processed_traj)

        self.replay_buffer.add(
            state=None,  # (n_envs, cond_steps, obs_dim)
            noise=None,  # (n_envs, horizon_steps, action_dim)
            action=None,  # (n_envs, horizon_steps, action_dim)
            reward=None,  # (n_envs, 1) - already has correct shape
            next_state=None,  # (n_envs, cond_steps, obs_dim)
            done=done_venv.unsqueeze(-1),  # (n_envs, 1)
            full_trajectory_info=full_trajectory_info,  # List of trajectory dicts from MultiStepFull
        )
    
    def _get_flow_action_for_eval(self, obs):
        """
        Override parent's method to handle image observations.
        Copy exact pretraining evaluation logic for perfect consistency.
        """
        with torch.no_grad():
            # Create condition dict exactly like pretraining - process all observation keys
            cond = {}
            state = obs["state"]  # (n_envs, cond_steps, obs_dim) or (n_envs, obs_dim)
            rgb = obs["rgb"]  # (n_envs, cond_steps, H, W, C) or (n_envs, H, W, C)
            
            # Convert to tensors
            if not isinstance(state, torch.Tensor):
                state = torch.from_numpy(state).float().to(self.device)
            if not isinstance(rgb, torch.Tensor):
                rgb = torch.from_numpy(rgb).float().to(self.device)
            
            # Ensure proper dimensions
            if state.ndim == 2:
                state = state.unsqueeze(1)  # Add cond_steps dim
            if rgb.ndim == 4:
                rgb = rgb.unsqueeze(1)  # Add cond_steps dim
            
            # Handle RGB format conversion (H, W, C) -> (C, H, W)
            if rgb.shape[-1] % 3 == 0:  # Channel last format
                rgb = rgb.permute(0, 1, 4, 2, 3)  # (n_envs, cond_steps, C, H, W)
            
            # Prepare condition dict for feature extraction
            cond = {
                'state': state,
                'rgb': rgb
            }

            # Use exact same model call as pretraining
            samples = self.model.pretrained_flow_policy(
                cond=cond
            )
            output_venv = samples.trajectories.cpu().numpy()  # n_env x horizon x act
            action_venv = output_venv[:, :self.act_steps]
            
            return action_venv
    