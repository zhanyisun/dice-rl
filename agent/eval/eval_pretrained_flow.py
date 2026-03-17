"""
Evaluation utilities for pretrained flow matching policies.

This module provides functions to evaluate flow matching policies before finetuning,
collecting trajectories and metrics for comparison with finetuned policies.
"""

import os
import torch
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from omegaconf import DictConfig

log = logging.getLogger(__name__)


def evaluate_flow_matching_policy(
    env,
    flow_policy,
    device: str = "cuda",
    n_episodes: int = 5,
    deterministic: bool = True,
    fixed_seeds: Optional[List[int]] = None,
    collect_trajectories: bool = True,
    horizon_steps: int = 8,
    act_steps: int = 8,
    cond_steps: int = 1,
    flow_steps: int = 10,
) -> Dict:
    """
    Evaluate a pretrained flow matching policy.
    
    Args:
        env: Vectorized environment
        flow_policy: Pretrained flow matching policy model
        device: Device to run on
        n_episodes: Number of evaluation episodes per environment
        deterministic: Whether to use deterministic actions
        fixed_seeds: Optional fixed seeds for reproducible evaluation
        collect_trajectories: Whether to collect state trajectories
        horizon_steps: Action horizon for the policy
        act_steps: Number of action steps to execute
        cond_steps: Number of conditioning steps
        flow_steps: Number of flow matching denoising steps
        
    Returns:
        Dictionary containing evaluation metrics and optionally trajectories
    """
    flow_policy.eval()
    n_envs = env.num_envs if hasattr(env, 'num_envs') else 1
    
    # Results storage
    all_episode_rewards = []
    all_episode_lengths = []
    all_trajectories = [] if collect_trajectories else None
    
    log.info(f"Evaluating flow matching policy for {n_episodes} episodes on {n_envs} environments")
    
    for episode in range(n_episodes):
        # Prepare options for reset with fixed seeds
        options_venv = [{} for _ in range(n_envs)]
        if fixed_seeds is not None:
            for env_idx in range(n_envs):
                # Use different seeds for each env, but consistent across evaluations
                seed = fixed_seeds[env_idx % len(fixed_seeds)] + 10000  # Offset for eval
                options_venv[env_idx]['seed'] = seed
        
        # Reset environments
        obs_venv = env.reset(options=options_venv) if hasattr(env, 'reset') else env.reset()
        
        episode_rewards = np.zeros(n_envs)
        episode_lengths = np.zeros(n_envs)
        episode_done = np.zeros(n_envs, dtype=bool)
        
        # Trajectory collection
        if collect_trajectories:
            env_trajectories = [[] for _ in range(n_envs)]
        
        # Run episode
        while not np.all(episode_done):
            # Collect states for trajectory
            if collect_trajectories:
                if isinstance(obs_venv, dict):
                    states = obs_venv['state']  # (n_envs, cond_steps, obs_dim)
                    if states.ndim == 3:
                        states = states[:, -1]  # Take last observation
                else:
                    states = obs_venv
                    if states.ndim == 3:
                        states = states[:, -1]
                
                for env_idx in range(n_envs):
                    if not episode_done[env_idx]:
                        env_trajectories[env_idx].append(states[env_idx].copy())
            
            # Get actions from flow matching policy
            with torch.no_grad():
                # Extract observation tensor
                if isinstance(obs_venv, dict):
                    obs_tensor = torch.from_numpy(obs_venv['state']).float().to(device)
                else:
                    obs_tensor = torch.tensor(obs_venv, dtype=torch.float32, device=device)
                
                # Ensure correct shape (n_envs, cond_steps, obs_dim)
                if obs_tensor.ndim == 2:
                    obs_tensor = obs_tensor.unsqueeze(1)  # Add cond_steps dimension
                
                # Generate actions using flow matching
                # Initialize noise for denoising process
                noise = torch.randn(n_envs, horizon_steps, flow_policy.action_dim).to(device)
                
                # Run flow matching denoising
                action = flow_policy(
                    cond=obs_tensor,
                    noise=noise,
                    num_steps=flow_steps,
                )  # (n_envs, horizon_steps, action_dim)
                
                # Extract action chunk to execute
                action_chunk = action[:, :act_steps].cpu().numpy()
            
            # Step environment
            obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = env.step(action_chunk)
            done_venv = terminated_venv | truncated_venv
            
            # Update episode statistics
            for env_idx in range(n_envs):
                if not episode_done[env_idx]:
                    episode_rewards[env_idx] += reward_venv[env_idx]
                    episode_lengths[env_idx] += 1
                    
                    if done_venv[env_idx]:
                        episode_done[env_idx] = True
        
        # Store results
        all_episode_rewards.extend(episode_rewards.tolist())
        all_episode_lengths.extend(episode_lengths.tolist())
        
        if collect_trajectories:
            for traj in env_trajectories:
                if len(traj) > 0:
                    all_trajectories.append(np.array(traj))
    
    # Compute statistics
    mean_reward = np.mean(all_episode_rewards)
    std_reward = np.std(all_episode_rewards)
    mean_length = np.mean(all_episode_lengths)
    success_rate = np.mean(np.array(all_episode_rewards) >= 1.0)  # Assuming binary success
    
    results = {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'mean_length': mean_length,
        'success_rate': success_rate,
        'all_rewards': all_episode_rewards,
        'all_lengths': all_episode_lengths,
    }
    
    if collect_trajectories:
        results['trajectories'] = all_trajectories
    
    log.info(f"Flow policy evaluation: reward={mean_reward:.3f}±{std_reward:.3f}, "
             f"length={mean_length:.1f}, success_rate={success_rate:.3f}")
    
    return results


def load_flow_matching_policy(checkpoint_path: str, device: str = "cuda"):
    """
    Load a pretrained flow matching policy from checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model on
        
    Returns:
        Loaded flow matching policy model
    """
    import hydra
    from model.flow_matching.flow_matching import FlowMatchingModel
    
    log.info(f"Loading flow matching policy from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model configuration if available
    if 'cfg' in checkpoint:
        model_cfg = checkpoint['cfg'].model
    else:
        # Default configuration
        model_cfg = DictConfig({
            'obs_dim': 59,  # For transport task
            'action_dim': 14,
            'noise_dim': 112,  # action_dim * horizon_steps
            'denoising_steps': 10,
            'hidden_dims': [512, 512, 512],
            'cond_dim': 59,
            'time_dim': 16,
            'flow_type': 'linear',
            'device': device,
        })
    
    # Create model
    model = hydra.utils.instantiate(model_cfg)
    
    # Load state dict
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model