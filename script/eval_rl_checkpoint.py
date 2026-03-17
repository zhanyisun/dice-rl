#!/usr/bin/env python3
"""
Evaluation script for RL checkpoints from agent/finetune/train_distill_flow_agent.py
Evaluates both the RL policy and the pretrained policy on test environments.
Handles both state-based and image-based models correctly.
"""

import argparse
import os
import math
import numpy as np
import torch
import logging
from pathlib import Path
from omegaconf import OmegaConf
import hydra
import pickle

from env.gym_utils import make_async

log = logging.getLogger(__name__)


def _extract_visual_features_from_obs(obs_venv, pretrained_model, device):
    """
    Extract visual features from raw observations using pretrained model.
    Copied from train_distill_flow_img_agent.py:_extract_visual_features_from_obs()
    
    Args:
        obs_venv: Dict with 'rgb' and 'state' keys from environment
        pretrained_model: The pretrained flow model for feature extraction
        device: Device to use
        
    Returns:
        Augmented state with visual features merged (as tensor)
    """
    # Get raw state and rgb from obs
    state = obs_venv["state"]  # (n_envs, cond_steps, obs_dim) or (n_envs, obs_dim)
    rgb = obs_venv["rgb"]  # (n_envs, cond_steps, H, W, C) or (n_envs, H, W, C)
    
    # Convert to tensors
    if not isinstance(state, torch.Tensor):
        state = torch.from_numpy(state).float().to(device)
    if not isinstance(rgb, torch.Tensor):
        rgb = torch.from_numpy(rgb).float().to(device)
    
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
        visual_features = pretrained_model.network.extract_visual_features(cond)
        # visual_features shape: (n_envs, feature_dim)
    
    # Merge visual features with original state
    # Repeat visual features for each cond_step
    if state.shape[1] > 1:  # Has cond_steps dimension
        visual_features = visual_features.unsqueeze(1).repeat(1, state.shape[1], 1)
    else:
        visual_features = visual_features.unsqueeze(1)

    augmented_state = torch.cat([state, visual_features], dim=-1)
    
    return augmented_state  # Return as tensor


def _get_flow_action_for_eval(model, obs, act_steps, device, use_image_obs):
    """
    Get action from pretrained flow policy for evaluation.
    Copied from train_distill_flow_agent.py and train_distill_flow_img_agent.py
    """
    with torch.no_grad():
        if use_image_obs:
            # Handle image observations (from train_distill_flow_img_agent.py)
            state = obs["state"]
            rgb = obs["rgb"]
            
            # Convert to tensors
            if not isinstance(state, torch.Tensor):
                state = torch.from_numpy(state).float().to(device)
            if not isinstance(rgb, torch.Tensor):
                rgb = torch.from_numpy(rgb).float().to(device)
            
            # Ensure proper dimensions
            if state.ndim == 2:
                state = state.unsqueeze(1)
            if rgb.ndim == 4:
                rgb = rgb.unsqueeze(1)
            
            # Handle RGB format conversion (H, W, C) -> (C, H, W)
            if rgb.shape[-1] % 3 == 0:
                rgb = rgb.permute(0, 1, 4, 2, 3)
            
            cond = {
                'state': state,
                'rgb': rgb
            }
        else:
            # Handle state-based observations
            if isinstance(obs, dict):
                obs_tensor = torch.from_numpy(obs['state']).float().to(device)
            else:
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
            
            if obs_tensor.ndim == 2:
                obs_tensor = obs_tensor.unsqueeze(1)
            
            cond = {"state": obs_tensor}
        
        # Use the pretrained flow policy
        sample = model.pretrained_flow_policy(
            cond=cond,
        )
        action = sample.trajectories
        
        return action[:, :act_steps].cpu().numpy()


def evaluate(model, eval_venv, num_eval_episodes, eval_n_envs, max_episode_steps, 
             act_steps, horizon_steps, device, strategy, 
             best_reward_threshold_for_success, render_video, render_dir, n_render,
             eval_type="rl", use_rl_policy=True, use_image_obs=False,
             pretrained_model=None):
    """
    Evaluate the agent - copied from train_distill_flow_agent.py:evaluate()
    with image observation handling from train_distill_flow_img_agent.py
    """
    model.eval()
    
    eval_episode_rewards = []
    eval_episode_lengths = []
    
    # Create fixed evaluation seeds
    eval_seeds = [seed for seed in range(10000, 10000 + num_eval_episodes * eval_n_envs)]
    
    # Run num_eval_episodes sequential episodes, each using eval_n_envs in parallel
    for episode in range(num_eval_episodes):
        # Prepare options with fixed seeds for reproducible evaluation
        options_venv = [{} for _ in range(eval_n_envs)]
        for env_idx in range(eval_n_envs):
            # Use fixed seeds for evaluation - different seeds for each episode+env combination
            seed_idx = episode * eval_n_envs + env_idx
            options_venv[env_idx]["seed"] = eval_seeds[seed_idx]
        
        if render_video:
            for env_ind in range(n_render):
                options_venv[env_ind]["video_path"] = os.path.join(
                    render_dir, f"{eval_type}_episode-{episode}_env-{env_ind}.mp4"
                )
        
        # Reset all environments for this episode with fixed seeds
        obs = eval_venv.reset_arg(options_list=options_venv)
        if isinstance(obs, list):
            obs = {
                key: np.stack([obs[i][key] for i in range(eval_n_envs)])
                for key in obs[0].keys()
            }
        
        episode_rewards = np.zeros(eval_n_envs)
        episode_lengths = np.zeros(eval_n_envs)
        episode_done = np.zeros(eval_n_envs, dtype=bool)
        
        step = 0
        # Run episode until all environments are done or max steps reached
        # Use ceiling division since we're executing act_steps actions at once
        max_eval_steps = math.ceil(max_episode_steps / act_steps)
        
        while step < max_eval_steps:
            # Get actions for active environments
            with torch.no_grad():
                if use_rl_policy:
                    # Use RL policy
                    if use_image_obs:
                        # Image-based: need to extract visual features first
                        augmented_state = _extract_visual_features_from_obs(
                            obs, pretrained_model, device
                        )
                    else:
                        # State-based: directly use state
                        if isinstance(obs, dict):
                            augmented_state = torch.from_numpy(obs["state"]).float().to(device)
                        else:
                            augmented_state = torch.tensor(obs, dtype=torch.float32, device=device)
                    
                    # Use exploration action method for non-standard strategies
                    action_venv, noise_venv = model.get_exploration_action(
                        augmented_state, 
                        num_samples=16,  # Fixed as requested
                        exploration_strategy=strategy,
                        training_step=1000000
                    )
                else:
                    # Use pretrained policy only
                    action_venv = _get_flow_action_for_eval(
                        model, obs, act_steps, device, use_image_obs
                    )
                    action_venv = torch.from_numpy(action_venv).to(device)
                
                action_chunk = action_venv[:, :act_steps]  # (n_envs, act_steps, action_dim)
            
            # Step environments
            obs, reward_venv, terminated_venv, truncated_venv, info_venv = eval_venv.step(action_chunk.cpu().numpy())
            step += 1
            done_venv = terminated_venv | truncated_venv
            
            # Update rewards and lengths for environments that were still active this step
            for env_idx in range(eval_n_envs):
                if not episode_done[env_idx]:
                    # Add reward and increment length for this step
                    episode_rewards[env_idx] += reward_venv[env_idx]
                    # Each chunk step represents act_steps actual environment steps
                    episode_lengths[env_idx] += act_steps
                    
                    # Check if this environment finished on this step
                    if done_venv[env_idx] or episode_lengths[env_idx] >= max_episode_steps:
                        episode_done[env_idx] = True
        
        # Collect results from all environments for this episode
        eval_episode_rewards.extend(episode_rewards.tolist())
        eval_episode_lengths.extend(episode_lengths.tolist())
    
    # Calculate metrics across all episodes
    mean_episode_reward = np.mean(eval_episode_rewards)
    std_episode_reward = np.std(eval_episode_rewards)
    mean_episode_length = np.mean(eval_episode_lengths)
    
    # Calculate success rate
    success_rate = np.mean(np.array(eval_episode_rewards) >= best_reward_threshold_for_success)
    
    log.info(f"Eval {eval_type}: "
            f"reward={mean_episode_reward:.3f}±{std_episode_reward:.3f}, "
            f"length={mean_episode_length:.1f}, "
            f"success_rate={success_rate:.3f} "
            f"({len(eval_episode_rewards)} episodes)")
    
    return {
        "return_mean": mean_episode_reward,
        "return_std": std_episode_reward,
        "length_mean": mean_episode_length,
        "success_rate": success_rate,
        "num_episodes": len(eval_episode_rewards),
    }


def main():
    
    parser = argparse.ArgumentParser(
        description="Evaluate RL checkpoint with both RL and pretrained policies"
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to RL checkpoint file (.pth or .pt)",
    )
    parser.add_argument(
        "--num_eval_episodes",
        type=int,
        default=10,
        help="Number of episodes to evaluate",
    )
    parser.add_argument(
        "--eval_n_envs",
        type=int,
        default=10,
        help="Number of parallel evaluation environments",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run evaluation on",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render videos of evaluation",
    )
    parser.add_argument(
        "--n_render",
        type=int,
        default=2,
        help="Number of environments to render",
    )
    
    args = parser.parse_args()
    
    # Fixed strategy settings
    strategy = "max_q_min"
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.ckpt_path}")
    checkpoint = torch.load(args.ckpt_path, map_location=args.device)
    
    # Extract directory path from checkpoint path and load config from .hydra/config.yaml
    ckpt_dir = os.path.dirname(args.ckpt_path)
    hydra_config_path = os.path.join(ckpt_dir, ".hydra", "config.yaml")
    
    if not os.path.exists(hydra_config_path):
        print(f"Warning: Hydra config not found at {hydra_config_path}, falling back to checkpoint's stored config")
        cfg = checkpoint["config"]
    else:
        print(f"Loading config from {hydra_config_path}")
        cfg = OmegaConf.load(hydra_config_path)
    
    # Override some config values for evaluation
    cfg.device = args.device
    
    # Determine if this is image-based or state-based
    use_image_obs = cfg.env.get("use_image_obs", False)
    
    print(f"Model type: {'Image-based' if use_image_obs else 'State-based'}")
    print(f"Environment: {cfg.env_name}")
    print(f"Evaluating with {args.eval_n_envs} parallel envs for {args.num_eval_episodes} episodes")
    print(f"Using strategy: {strategy}")
    
    # For image-based models, we need to update obs_dim to include visual features
    # This follows what train_distill_flow_img_agent.py and train_distill_residual_flow_img_agent.py do
    if use_image_obs:
        original_obs_dim = cfg.obs_dim  # Save original (should be 9 for robomimic tasks)
        
        # Get base_policy_path from config and load pretrained model to get visual_feature_dim
        if hasattr(cfg, 'base_policy_path'):
            print(f"Loading pretrained model config from {cfg.base_policy_path}")
            
            # The pretrained config is in .hydra/config.yaml, not in the checkpoint itself
            # Extract the directory path from the checkpoint path
            pretrained_dir = os.path.dirname(os.path.dirname(cfg.base_policy_path))  # Go up from checkpoint/ to main dir
            pretrained_config_path = os.path.join(pretrained_dir, ".hydra", "config.yaml")
            
            if os.path.exists(pretrained_config_path):
                print(f"  Loading pretrained config from {pretrained_config_path}")
                pretrained_config = OmegaConf.load(pretrained_config_path)
            else:
                # Fallback: try loading from checkpoint in case it's stored there
                print(f"  Config not found at {pretrained_config_path}, checking checkpoint...")
                pretrained_checkpoint = torch.load(cfg.base_policy_path, map_location='cpu')
                if 'cfg' in pretrained_checkpoint:
                    pretrained_config = pretrained_checkpoint['cfg']
                elif 'config' in pretrained_checkpoint:
                    pretrained_config = pretrained_checkpoint['config']
                else:
                    raise ValueError(f"No config found at {pretrained_config_path} or in checkpoint at {cfg.base_policy_path}")
            
            # Extract visual feature dimension based on network type
            network_target = pretrained_config.model.network._target_
            
            # For ResNet models: use visual_feature_dim directly
            visual_feature_dim = pretrained_config.model.network.visual_feature_dim
            model_type = "VisionResNetDiffusionMLP" if "Diffusion" in network_target else "VisionResNetFlowMatchingMLP"
            print(f"  Using {model_type} visual_feature_dim: {visual_feature_dim}")

            # Update obs_dim to include visual features
            cfg.obs_dim = visual_feature_dim + original_obs_dim
            print(f"Image-based model obs_dim update:")
            print(f"  Original obs_dim: {original_obs_dim}")
            print(f"  Visual feature dim: {visual_feature_dim}")
            print(f"  Updated obs_dim: {cfg.obs_dim} (for critic input)")
            print(f"  With action history: {cfg.obs_dim + cfg.action_dim * cfg.horizon_steps} (full critic input)")
        else:
            print("Warning: base_policy_path not found in config for image-based model")
            print("Attempting to use obs_dim from config as-is (may cause errors)")
    
    # Create evaluation environments BEFORE model (following train_agent.py pattern)
    print("Creating evaluation environments...")
    env_type = cfg.env.get("env_type", None)
    
    # Setup render directory if needed
    if args.render:
        render_dir = os.path.join(os.path.dirname(args.ckpt_path), "eval_render")
        os.makedirs(render_dir, exist_ok=True)
        print(f"Videos will be saved to {render_dir}")
    else:
        render_dir = None
    
    print(f"Using obs_dim={cfg.obs_dim} for {'image-based' if use_image_obs else 'state-based'} environment creation")
    
    eval_venv = make_async(
        cfg.env.name,
        env_type=env_type,
        num_envs=args.eval_n_envs,
        asynchronous=True,
        max_episode_steps=cfg.env.max_episode_steps,
        wrappers=cfg.env.get("wrappers", None),
        robomimic_env_cfg_path=cfg.get("robomimic_env_cfg_path", None),
        shape_meta=cfg.get("shape_meta", None),
        use_image_obs=use_image_obs,
        render=False,
        render_offscreen=args.render,
        obs_dim=cfg.obs_dim,  # Use the config's obs_dim directly (already augmented for image-based)
        action_dim=cfg.action_dim,
        **cfg.env.specific if "specific" in cfg.env else {},
    )
    
    # Seed environments if not furniture (following train_agent.py pattern)
    if env_type != "furniture":
        initial_seeds = [10000 + i for i in range(args.eval_n_envs)]
        eval_venv.seed(initial_seeds)
        print(f"Seeded environments with initial seeds: {initial_seeds}")
    
    # Now initialize model AFTER environments (following train_agent.py pattern)
    print("Initializing model...")
    model = hydra.utils.instantiate(cfg.model)
    
    # Load model weights (use strict=False to ignore unexpected keys like confidence tracking)
    print("Loading model weights...")
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    
    if missing_keys:
        print(f"Warning: Missing keys in checkpoint: {missing_keys}")
    if unexpected_keys:
        print(f"Note: Ignoring unexpected keys in checkpoint (likely removed during cleanup): {unexpected_keys}")
    
    model.to(args.device)
    model.eval()
    
    # For image-based models, get the pretrained model for feature extraction
    pretrained_model = None
    if use_image_obs:
        pretrained_model = model.pretrained_flow_policy
        pretrained_model.eval()

    horizon_steps = cfg.horizon_steps
    act_steps = cfg.act_steps
    max_episode_steps = cfg.env.max_episode_steps
    best_reward_threshold_for_success = cfg.env.get("best_reward_threshold_for_success", 1.0)
    
    print("\n" + "="*60)
    print("Evaluating RL Policy (Pretrained + Residual)")
    print("="*60)
    
    rl_metrics = evaluate(
        model=model,
        eval_venv=eval_venv,
        num_eval_episodes=args.num_eval_episodes,
        eval_n_envs=args.eval_n_envs,
        max_episode_steps=max_episode_steps,
        act_steps=act_steps,
        horizon_steps=horizon_steps,
        device=args.device,
        strategy=strategy,
        best_reward_threshold_for_success=best_reward_threshold_for_success,
        render_video=args.render,
        render_dir=render_dir,
        n_render=args.n_render,
        eval_type="rl_policy",
        use_rl_policy=True,
        use_image_obs=use_image_obs,
        pretrained_model=pretrained_model,
    )
    
    print(f"RL Policy Results:")
    print(f"  Return: {rl_metrics['return_mean']:.2f} ± {rl_metrics['return_std']:.2f}")
    print(f"  Success Rate: {rl_metrics['success_rate']*100:.1f}%")
    
    print("\n" + "="*60)
    print("Evaluating Pretrained Policy Only")
    print("="*60)
    
    pretrained_metrics = evaluate(
        model=model,
        eval_venv=eval_venv,
        num_eval_episodes=args.num_eval_episodes,
        eval_n_envs=args.eval_n_envs,
        max_episode_steps=max_episode_steps,
        act_steps=act_steps,
        horizon_steps=horizon_steps,
        device=args.device,
        strategy="standard",
        best_reward_threshold_for_success=best_reward_threshold_for_success,
        render_video=args.render,
        render_dir=render_dir,
        n_render=3,
        eval_type="pretrained_policy",
        use_rl_policy=False,
        use_image_obs=use_image_obs,
        pretrained_model=pretrained_model,
    )
    
    print(f"Pretrained Policy Results:")
    print(f"  Return: {pretrained_metrics['return_mean']:.2f} ± {pretrained_metrics['return_std']:.2f}")
    print(f"  Success Rate: {pretrained_metrics['success_rate']*100:.1f}%")
    
    print("\n" + "="*60)
    print("Comparison")
    print("="*60)
    
    return_improvement = rl_metrics['return_mean'] - pretrained_metrics['return_mean']
    success_improvement = (rl_metrics['success_rate'] - pretrained_metrics['success_rate']) * 100
    
    print(f"Return Improvement: {return_improvement:+.2f}")
    print(f"Success Rate Improvement: {success_improvement:+.1f}%")
    
    # Save results
    results = {
        "ckpt_path": args.ckpt_path,
        "num_eval_episodes": args.num_eval_episodes,
        "eval_n_envs": args.eval_n_envs,
        "rl_policy": rl_metrics,
        "pretrained_policy": pretrained_metrics,
        "improvement": {
            "return": return_improvement,
            "success_rate": success_improvement,
        },
    }
    
    # Save to file
    result_path = args.ckpt_path.replace(".pth", "").replace(".pt", "") + "_eval_results.pkl"
    with open(result_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {result_path}")
    
    # Close environments
    eval_venv.close()


if __name__ == "__main__":
    main()