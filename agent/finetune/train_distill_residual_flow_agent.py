"""
Finetune a one-step distilled actor with online RL
"""

import os
import pickle
import einops
import numpy as np
import torch
import logging
import wandb
import hydra
import math
import csv
from typing import Dict, List, Optional

log = logging.getLogger(__name__)
from util.timer import Timer
from agent.finetune.train_agent import TrainAgent
from util.scheduler import CosineAnnealingWarmupRestarts
from util.hybrid_replay_buffer import HybridReplayBuffer
from util.state_visualization import StateTrajectoryVisualizer
from env.gym_utils import make_async

class TrainDistillResidualFlowAgent(TrainAgent):
    
    def __init__(self, cfg):
        cfg = self._setup_upsampling_wrapper(cfg)
        
        super().__init__(cfg)

        self.noise_dim = cfg.get("noise_dim", self.action_dim)
        self.horizon_steps = cfg.get("horizon_steps", 8)
        self.cond_steps = cfg.get("cond_steps", 1)
        
        # Adaptive expert ratio settings
        self.use_adaptive_expert_ratio = cfg.get("use_adaptive_expert_ratio", False)
        self.adaptive_expert_ratio_start = cfg.get("adaptive_expert_ratio_start", 1.0)
        self.adaptive_expert_ratio_end = cfg.get("adaptive_expert_ratio_end", 0.1)
        self.adaptive_expert_ratio_steps = cfg.get("adaptive_expert_ratio_steps", 40000)

        # Set up replay buffer
        self._setup_replay_buffer(cfg)
        
        # Training parameters
        self.num_train_steps = cfg.train.get("num_train_steps", 100000)
        self.eval_freq = cfg.train.get("eval_freq", 1000)
        self.gamma = cfg.train.get("gamma", 0.99)
        self.tau = cfg.train.get("tau", 0.005)  # Target network update rate
        self.update_freq = cfg.train.get("update_freq", 1)
        self.gradient_steps = cfg.train.get("gradient_steps", 1)

        self.max_n_episodes = cfg.expert_dataset.get("max_n_episodes", 50)
        
        # Delayed update frequencies (like CURL/SAC)
        self.actor_update_freq = cfg.train.get("actor_update_freq", 1)  # Update actor every N steps
        self.critic_target_update_freq = cfg.train.get("critic_target_update_freq", 1)  # Update target every N steps

        # Optimizers
        # Create base Adam optimizer first
        self.actor_optimizer = torch.optim.Adam(
            self.model.actor.parameters(),
            lr=cfg.train.actor_lr,
            weight_decay=cfg.train.get("actor_weight_decay", 0.0),
        )
        
        self.critic_optimizer = torch.optim.Adam(
            self.model.critic.parameters(),
            lr=cfg.train.critic_lr,
            weight_decay=cfg.train.get("critic_weight_decay", 0.0),
        )

        # Learning rate schedulers
        if cfg.train.get("use_lr_scheduler", False):
            self.actor_scheduler = CosineAnnealingWarmupRestarts(
                self.actor_optimizer,
                first_cycle_steps=cfg.train.lr_scheduler.first_cycle_steps,
                cycle_mult=1.0,
                max_lr=cfg.train.actor_lr,
                min_lr=cfg.train.lr_scheduler.min_lr,
                warmup_steps=cfg.train.lr_scheduler.warmup_steps,
                gamma=1.0,
            )
            
            self.critic_scheduler = CosineAnnealingWarmupRestarts(
                self.critic_optimizer,
                first_cycle_steps=cfg.train.lr_scheduler.first_cycle_steps,
                cycle_mult=1.0,
                max_lr=cfg.train.critic_lr,
                min_lr=cfg.train.lr_scheduler.min_lr,
                warmup_steps=cfg.train.lr_scheduler.warmup_steps,
                gamma=1.0,
            )
        else:
            self.actor_scheduler = None
            self.critic_scheduler = None
        
        # Exploration strategy settings
        self.online_explore_strategy = cfg.get("online_explore_strategy", "standard")
        self.evaluate_strategy = cfg.get("evaluate_strategy", "standard")
        self.num_exploration_samples = cfg.get("num_exploration_samples", 10)
        
        # Track training steps for exploration warmup
        self.current_training_step = 0
        
        log.info(f"  Collecting transitions at all timesteps (not just t=0, {self.horizon_steps}, {2*self.horizon_steps}, ...)")
        log.info(f"  Sampling uniformly from all timesteps for training")

        # Logging
        self.log_freq = cfg.train.get("log_freq", 100)
        self.save_freq = cfg.train.get("save_freq", 1000)
        self.log_q_overestimation = cfg.get("log_q_overestimation", False)
        
        # Visualization settings
        self.visualize_trajectories = cfg.get("visualize_trajectories", True)
        self.visualization_freq = cfg.get("visualization_freq", 5000)  # How often to create visualizations
        self.num_eval_episodes = cfg.get("num_eval_episodes", 5)  # Number of eval episodes to visualize
        # Fixed seeds for evaluation: num_eval_episodes * n_envs total seeds
        self.eval_seeds = list(range(10000, 10000 + self.num_eval_episodes * self.eval_n_envs))
        
        # Initialize trajectory visualizer
        if self.visualize_trajectories:
            self.trajectory_visualizer = StateTrajectoryVisualizer()
            self.trajectory_data = {}  # Store trajectories for comparison
        
        log.info(f"TrainDistillResidualFlowAgent initialized with:")
        log.info(f"  num_train_steps: {self.num_train_steps}")
        log.info(f"  eval_freq: {self.eval_freq}")
        log.info(f"  noise_dim: {self.noise_dim}")
        log.info(f"  horizon_steps: {self.horizon_steps}")
        log.info(f"  cond_steps: {self.cond_steps}")
        log.info(f"  gamma: {self.gamma}")
        log.info(f"  tau: {self.tau}")
        log.info(f"  actor_update_freq: {self.actor_update_freq}")
        log.info(f"  critic_target_update_freq: {self.critic_target_update_freq}")
        # log.info(f"  bc_loss_weight: {self.bc_loss_weight}")
        
        # Initialize CSV file for evaluation results
        self.eval_csv_path = os.path.join(self.logdir, "evaluation_results.csv")
        self._init_eval_csv()
    
    def _init_eval_csv(self):
        """Initialize CSV file with headers for evaluation results."""
        with open(self.eval_csv_path, 'w', newline='') as csvfile:
            fieldnames = ['step', 'eval_type', 'mean_reward', 'std_reward', 'success_rate', 'mean_length', 'num_episodes']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        log.info(f"Initialized evaluation CSV at: {self.eval_csv_path}")
    
    def _log_eval_to_csv(self, step, eval_type, mean_reward, std_reward, success_rate, mean_length=None, num_episodes=None):
        """Append evaluation results to CSV file."""
        with open(self.eval_csv_path, 'a', newline='') as csvfile:
            fieldnames = ['step', 'eval_type', 'mean_reward', 'std_reward', 'success_rate', 'mean_length', 'num_episodes']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({
                'step': step,
                'eval_type': eval_type,
                'mean_reward': mean_reward,
                'std_reward': std_reward,
                'success_rate': success_rate,
                'mean_length': mean_length if mean_length is not None else '',
                'num_episodes': num_episodes if num_episodes is not None else ''
            })
    
    def _setup_replay_buffer(self, cfg):
        """
        Set up replay buffer with expert dataset for RLPD.
        Can be overridden by child classes for custom preprocessing.
        """
        # Initialize expert dataset for RLPD if enabled
        expert_dataset = None
        if cfg.get("use_rlpd", False) and hasattr(cfg, 'expert_dataset'):
            expert_dataset = hydra.utils.instantiate(cfg.expert_dataset)
            log.info(f"RLPD enabled: loaded expert dataset with {len(expert_dataset)} transitions")

        # Create replay buffer with shared arguments
        self.replay_buffer = HybridReplayBuffer(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            noise_dim=self.noise_dim,
            max_size=cfg.replay_buffer.get("max_size", 1000000),
            n_envs=self.n_envs,  # Number of parallel environments
            cond_steps=self.cond_steps,  # Number of observation history steps
            horizon_steps=self.horizon_steps,  # Number of action horizon steps
            device=self.device,
            gamma=cfg.train.get("gamma", 0.99),  # Discount factor for MC return computation
            log_q_overestimation=cfg.get("log_q_overestimation", False),  # Whether to compute MC returns
            # RLPD settings
            use_rlpd=cfg.get("use_rlpd", False),
            expert_ratio=cfg.get("expert_ratio", 0.5),
            expert_dataset=expert_dataset,
            # N-step returns
            use_n_step=cfg.replay_buffer.get("use_n_step", False),
            n_step=cfg.replay_buffer.get("n_step", 1),
            # Expert dataset n-step settings (can differ from online data)
            expert_use_n_step=cfg.replay_buffer.get("expert_use_n_step", False),
            expert_n_step=cfg.replay_buffer.get("expert_n_step", 1),
        )
    
    def get_current_expert_ratio(self, env_steps):
        """
        Calculate the current expert ratio based on environment steps.
        
        Args:
            env_steps: Current number of environment steps since training started
            
        Returns:
            expert_ratio: Current expert ratio for RLPD sampling
        """
        if not self.use_adaptive_expert_ratio:
            # Use fixed expert ratio from config
            return self.cfg.get("expert_ratio", 0.5)
        
        # Linear decay from start to end over specified steps
        if env_steps >= self.adaptive_expert_ratio_steps:
            return self.adaptive_expert_ratio_end
        
        # Linear interpolation
        progress = env_steps / self.adaptive_expert_ratio_steps
        current_ratio = self.adaptive_expert_ratio_start - (self.adaptive_expert_ratio_start - self.adaptive_expert_ratio_end) * progress
        
        return current_ratio
    
    def collect_transition(self, obs_venv, action_venv, noise_venv, reward_venv, done_venv, next_obs_venv, info_venv=None):
        """
        Collect a transition and add it to the replay buffer.
        
        Args:
            obs_venv: dict with 'state' key or tensor - chunked current observations
            action_venv: (n_envs, horizon_steps, action_dim) - chunked actions taken
            noise_venv: (n_envs, horizon_steps, action_dim) - chunked noise used to generate actions
            reward_venv: (n_envs, 1) - rewards received (already has correct shape)
            done_venv: (n_envs,) - done flags  
            next_obs_venv: dict with 'state' key or tensor - chunked next observations
            info_venv: Optional[List[Dict]] - info dict from environment step with full trajectory data
        """
        # Extract state from dictionary observations if needed
        full_trajectory_info = []
        for env_idx in range(self.n_envs):
            traj = info_venv[env_idx]['full_trajectory']
            
            # Copy trajectory structure and extract state from observations
            processed_traj = {
                'initial_obs': None,
                'observations': [],
                'actions': traj['actions'],
                'rewards': traj['rewards'],
                'dones': traj['dones'],
                'include_initial': traj['include_initial'],
            }
            
            # Process initial observation if present
            if traj['include_initial'] and traj['initial_obs'] is not None:
                if isinstance(traj['initial_obs'], dict):
                    processed_traj['initial_obs'] = torch.from_numpy(traj['initial_obs']["state"]).float().to(self.device)
                else:
                    processed_traj['initial_obs'] = traj['initial_obs']
            
            # Process all observations - extract state from dict
            for obs in traj['observations']:
                if isinstance(obs, dict):
                    state_tensor = torch.from_numpy(obs["state"]).float().to(self.device)
                else:
                    state_tensor = obs
                processed_traj['observations'].append(state_tensor)
            
            full_trajectory_info.append(processed_traj)
        
        # For upsampling, we only need to pass the full_trajectory_info
        # state and next_state are not used by HybridReplayBuffer in upsampling mode
        self.replay_buffer.add(
            state=None,  # Not used in upsampling mode
            noise=None,  # Not used in upsampling mode
            action=None,  # Not used in upsampling mode
            reward=None,  # Not used in upsampling mode
            next_state=None,  # Not used in upsampling mode
            done=done_venv.unsqueeze(-1),  # (n_envs, 1)
            full_trajectory_info=full_trajectory_info  # List of trajectory dicts from MultiStepFull
        )
    
    def update_networks(self, training_step: int = 0):
        """
        Update all networks using data from replay buffer.
        
        Args:
            training_step: Current training step for Q-filtering warm-up
        """
        if self.replay_buffer.get_total_transitions() < self.batch_size:
            return
        
        # Get current expert ratio based on total steps (for adaptive scheduling)
        current_expert_ratio = self.get_current_expert_ratio(training_step)
        
        # Sample batch from replay buffer with current expert ratio
        if self.log_q_overestimation:
            state, noise, action, reward, next_state, done, mc_return, n_steps, data_source = self.replay_buffer.sample(
                self.batch_size, 
                expert_ratio=current_expert_ratio
            )
        else:
            state, noise, action, reward, next_state, done, n_steps, data_source = self.replay_buffer.sample(
                self.batch_size, 
                expert_ratio=current_expert_ratio
            )
        
        # Q-value overestimation analysis (compute before loss for Q-filtering)
        q_overestimation = None
        if self.log_q_overestimation:
            with torch.no_grad():
                predicted_q = self.model.critic(state, noise=None, action=action)  # (batch_size, 1)
                
                # Compare with ground truth MC returns
                q_overestimation = predicted_q - mc_return  # (batch_size, 1)
        
        # regenerate noise
        noise = torch.randn(state.shape[0], self.horizon_steps, self.action_dim, device=self.device)
        # Compute losses
        loss_dict = self.model.loss(
            state=state,
            noise=noise,
            action=action,
            next_state=next_state,
            reward=reward,
            done=done,
            gamma=self.gamma,
            training_step=training_step,
            q_overestimation=q_overestimation,  # Pass q_overestimation for Q-filtering
            n_steps=n_steps,  # Pass n_steps for per-sample gamma computation
            data_source=data_source,  # Pass data source labels (0=online, 1=expert)
        )
        
        # Add Q-analysis metrics if computed
        if self.log_q_overestimation and q_overestimation is not None:
            # Separate rewards for online and offline data
            online_mask = (data_source == 0).float()  # (B, 1) - 1.0 for online, 0.0 for expert
            offline_mask = (data_source == 1).float()  # (B, 1) - 1.0 for expert, 0.0 for online
            
            # Calculate online and offline rewards separately
            online_count = online_mask.sum()
            offline_count = offline_mask.sum()
            
            online_reward = (reward * online_mask).sum() / online_count if online_count > 0 else 0.0
            offline_reward = (reward * offline_mask).sum() / offline_count if offline_count > 0 else 0.0
            
            loss_dict.update({
                'q_analysis/overestimation_mean': q_overestimation.mean().item(),
                'q_analysis/predicted_q_mean': predicted_q.mean().item(),
                'q_analysis/ground_truth_q_mean': mc_return.mean().item(),
                'q_analysis/reward': reward.mean().item(),
                'q_analysis/online_reward': online_reward.item() if torch.is_tensor(online_reward) else online_reward,
                'q_analysis/offline_reward': offline_reward.item() if torch.is_tensor(offline_reward) else offline_reward,
            })
        
        # Update actor (with delayed updates)
        if (training_step+1) % self.actor_update_freq == 0:
            self.actor_optimizer.zero_grad()
            loss_dict['actor_total'].backward()
            
            # Compute actor gradient norm before clipping
            actor_grad_norm = 0.0
            for param in self.model.actor.parameters():
                if param.grad is not None:
                    actor_grad_norm += param.grad.data.norm(2).item() ** 2
            actor_grad_norm = actor_grad_norm ** 0.5
            loss_dict['actor_grad_norm'] = actor_grad_norm
            
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
            
        else:
            # Clean up computation graph even if actor not updated
            loss_dict['actor_total'].backward()
            loss_dict['actor_grad_norm'] = 0.0  # No gradient computed
            
        # Update critic (always update)
        self.critic_optimizer.zero_grad()
        loss_dict['critic_loss'].backward()
        
        # Compute critic gradient norm before clipping
        critic_grad_norm = 0.0
        for param in self.model.critic.parameters():
            if param.grad is not None:
                critic_grad_norm += param.grad.data.norm(2).item() ** 2
        critic_grad_norm = critic_grad_norm ** 0.5
        loss_dict['critic_grad_norm'] = critic_grad_norm
        
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()
        
        # Update target networks with delayed updates (like CURL/SAC)
        if (training_step+1) % self.critic_target_update_freq == 0:
            self.model.update_target_networks(tau=self.tau)
        
        # Update learning rate schedulers (only when actor updates)
        if (training_step+1) % self.actor_update_freq == 0:
            if self.actor_scheduler is not None:
                self.actor_scheduler.step()
        if self.critic_scheduler is not None:
            self.critic_scheduler.step()
        
        # Add update frequency metrics to loss_dict for logging
        loss_dict['actor_updated'] = float(training_step % self.actor_update_freq == 0)
        loss_dict['target_updated'] = float(training_step % self.critic_target_update_freq == 0)

        return loss_dict, current_expert_ratio
    
    def get_action(self, obs_venv, strategy="standard"):
        """
        Get actions from the distilled actor using the specified strategy.
        
        Args:
            obs_venv: dict with 'state' key or (n_envs, cond_steps, obs_dim) - chunked observations
            strategy: str - name of the strategy to use (e.g., "standard", "max_q_min", "max_q_std")
            
        Returns:
            action_venv: (n_envs, horizon_steps, action_dim) - chunked actions
            noise_venv: (n_envs, horizon_steps, action_dim) - chunked noise used to generate actions
        """
        with torch.no_grad():
            # Handle dictionary observations (robomimic format)
            if isinstance(obs_venv, dict):
                # Extract state from dictionary observations
                state_venv = torch.from_numpy(obs_venv["state"]).float().to(self.device)  # (n_envs, cond_steps, obs_dim)
            else:
                # Convert numpy observations to tensor
                state_venv = torch.tensor(obs_venv, dtype=torch.float32, device=self.device)  # (n_envs, cond_steps, obs_dim)
            
            # Generate actions based on the specified strategy
            if strategy != "standard":
                # Use exploration action method for non-standard strategies
                action_venv, noise_venv = self.model.get_exploration_action(
                    state_venv, 
                    num_samples=self.num_exploration_samples,
                    exploration_strategy=strategy,
                    training_step=self.current_training_step  # Use current training step
                )
            else:
                # Standard action generation (single sample with random noise)
                noise_venv = torch.randn(self.n_envs, self.horizon_steps, self.action_dim, device=self.device)
                action_venv = self.model.get_action(state_venv, noise_venv)

        return action_venv, noise_venv
    
    def evaluate(self, total_steps: int, collect_trajectories: bool = False):
        """
        Evaluate the agent and log metrics using all parallel environments.
        
        Args:
            total_steps: Current training step
            collect_trajectories: Whether to collect trajectories for visualization
        """
        self.model.eval()
        
        eval_episode_rewards = []
        eval_episode_lengths = []
        eval_trajectories = [] if collect_trajectories else None
        
        # Run num_eval_episodes sequential episodes, each using eval_n_envs in parallel
        for episode in range(self.num_eval_episodes):
            # Prepare options with fixed seeds for reproducible evaluation
            options_venv = [{} for _ in range(self.eval_n_envs)]
            for env_idx in range(self.eval_n_envs):
                # Use fixed seeds for evaluation - different seeds for each episode+env combination
                seed_idx = episode * self.eval_n_envs + env_idx
                options_venv[env_idx]["seed"] = self.eval_seeds[seed_idx]
            
            if self.render_video:
                for env_ind in range(self.n_render):
                    options_venv[env_ind]["video_path"] = os.path.join(
                        self.render_dir, f"{total_steps}_eval_episode-{episode}_env-{env_ind}.mp4"
                    )
            
            # Reset all EVAL environments for this episode with fixed seeds
            obs = self.reset_env_all(options_venv=options_venv, eval_flag=True)
            episode_rewards = np.zeros(self.eval_n_envs)
            episode_lengths = np.zeros(self.eval_n_envs)
            episode_done = np.zeros(self.eval_n_envs, dtype=bool)
            
            # Trajectory collection
            if collect_trajectories:
                env_trajectories = [[] for _ in range(self.eval_n_envs)]
            
            step = 0
            # Run episode until all environments are done or max steps reached
            # Use ceiling division since we're executing act_steps actions at once
            max_eval_steps = math.ceil(self.max_episode_steps / self.act_steps)
            while step < max_eval_steps:
                # Collect only robot end-effector positions for trajectory visualization (save memory)
                if collect_trajectories:
                    if isinstance(obs, dict):
                        states = obs['state']  # (n_envs, cond_steps, obs_dim)
                        if states.ndim == 3:
                            states = states[:, -1]  # Take last observation
                    else:
                        states = obs
                        if states.ndim == 3:
                            states = states[:, -1]
                    
                    for env_idx in range(self.eval_n_envs):
                        if not episode_done[env_idx]:
                            # Only save robot0_eef_pos [0:3] and robot1_eef_pos [9:12]
                            eef_state = np.concatenate([states[env_idx][:3], states[env_idx][9:12]])
                            env_trajectories[env_idx].append(eef_state)
                
                # Get actions using evaluation strategy for active environments
                with torch.no_grad():
                    action_venv, noise_venv = self.get_action(obs, strategy=self.evaluate_strategy)
                    action_chunk = action_venv[:, : self.act_steps]  # (n_envs, act_steps, action_dim)
                
                # Step EVAL environments (reset_within_step=True handles auto-reset for finished envs)
                obs, reward_venv, terminated_venv, truncated_venv, info_venv = self.eval_venv.step(action_chunk.cpu().numpy())
                step += 1
                done_venv = terminated_venv | truncated_venv

                # Update rewards and lengths for environments that were still active this step
                for env_idx in range(self.eval_n_envs):
                    if not episode_done[env_idx]:
                        # Add reward and increment length for this step
                        episode_rewards[env_idx] += reward_venv[env_idx]
                        # Each chunk step represents act_steps actual environment steps
                        episode_lengths[env_idx] += self.act_steps
                        
                        # Check if this environment finished on this step
                        if done_venv[env_idx] or episode_lengths[env_idx] >= self.max_episode_steps:
                            episode_done[env_idx] = True
                            # Note: With reset_within_step=True, the environment will auto-reset
                            # for the next episode, but we ignore further steps for this episode
            
            # Collect results from all environments for this episode
            eval_episode_rewards.extend(episode_rewards.tolist())
            eval_episode_lengths.extend(episode_lengths.tolist())
            
            # Store trajectories
            if collect_trajectories:
                for env_idx, traj in enumerate(env_trajectories):
                    if len(traj) > 0:
                        traj_array = np.array(traj)
                        eval_trajectories.append(traj_array)

        # Calculate metrics across all episodes (num_eval_episodes * n_envs total)
        mean_episode_reward = np.mean(eval_episode_rewards)
        std_episode_reward = np.std(eval_episode_rewards)
        mean_episode_length = np.mean(eval_episode_lengths)
        
        # Calculate success rate
        success_rate = np.mean(np.array(eval_episode_rewards) >= self.best_reward_threshold_for_success)
        
        # Log evaluation metrics
        if self.use_wandb:
            wandb.log({
                'eval/episode_reward': mean_episode_reward,
                'eval/episode_reward_std': std_episode_reward, 
                'eval/episode_length': mean_episode_length,
                'eval/success_rate': success_rate,
                'eval/num_episodes': len(eval_episode_rewards),
            }, step=total_steps*self.horizon_steps*self.n_envs)
        
        log.info(f"Eval at step {total_steps}: "
                f"reward={mean_episode_reward:.3f}±{std_episode_reward:.3f}, "
                f"length={mean_episode_length:.1f}, "
                f"success_rate={success_rate:.3f} "
                f"({len(eval_episode_rewards)} episodes)")
        
        # Log evaluation results to CSV
        actual_step = total_steps * self.horizon_steps * self.n_envs
        self._log_eval_to_csv(
            step=actual_step,
            eval_type='finetuned',
            mean_reward=mean_episode_reward,
            std_reward=std_episode_reward,
            success_rate=success_rate,
            mean_length=mean_episode_length,
            num_episodes=len(eval_episode_rewards)
        )
        
        # Switch back to training mode
        self.model.train()
        self.model.pretrained_flow_policy.eval()
        
        # Note: Main training loop will reset environments after evaluation
        
        if collect_trajectories:
            return mean_episode_reward, success_rate, eval_trajectories
        else:
            return mean_episode_reward, success_rate
    
    def _get_flow_action_for_eval(self, obs):
        """
        Get action from pretrained flow policy for evaluation.
        This is separated out so child classes can override it.
        """
        with torch.no_grad():
            if isinstance(obs, dict):
                obs_tensor = torch.from_numpy(obs['state']).float().to(self.device)
            else:
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
            
            if obs_tensor.ndim == 2:
                obs_tensor = obs_tensor.unsqueeze(1)
            
            # Use the pretrained flow policy - FlowMatchingModel expects dict with "state" key
            sample = self.model.pretrained_flow_policy(
                cond={"state": obs_tensor},
            )
            action = sample.trajectories
            
            return action[:, :self.act_steps].cpu().numpy()
    
    def evaluate_pretrained_flow_policy(self):
        """
        Evaluate the pretrained flow matching policy before finetuning.
        Uses exact same seeds and episodes as regular evaluation.
        """
        log.info("Evaluating pretrained flow matching policy...")
        self.model.eval()
        
        # Collect trajectories with same seeds as regular evaluation
        trajectories = []
        rewards = []
        reward_trajs_all = []  # To store per-timestep rewards like pretraining
        
        # Use same number of episodes as regular evaluation
        for episode in range(self.num_eval_episodes):
            # Use exact same seed pattern as regular evaluation
            options_venv = [{} for _ in range(self.eval_n_envs)]
            for env_idx in range(self.eval_n_envs):
                seed_idx = episode * self.eval_n_envs + env_idx
                options_venv[env_idx]["seed"] = self.eval_seeds[seed_idx]
            
            obs = self.reset_env_all(options_venv=options_venv, eval_flag=True)
            episode_done = np.zeros(self.eval_n_envs, dtype=bool)
            env_trajectories = [[] for _ in range(self.eval_n_envs)]
            env_rewards = np.zeros(self.eval_n_envs)
            reward_trajs = []  # Store per-timestep rewards like pretraining
            step_count = 0
            # Use ceiling division since we're executing act_steps actions at once
            max_eval_steps = math.ceil(self.max_episode_steps / self.act_steps)
            while step_count < max_eval_steps:
                # Collect only robot end-effector positions (save memory)
                if isinstance(obs, dict):
                    states = obs['state']
                    if states.ndim == 3:
                        states = states[:, -1]
                else:
                    states = obs
                    if states.ndim == 3:
                        states = states[:, -1]
                
                for env_idx in range(self.eval_n_envs):
                    if not episode_done[env_idx]:
                        # Only save robot0_eef_pos [0:3] and robot1_eef_pos [9:12]
                        eef_state = np.concatenate([states[env_idx][:3], states[env_idx][9:12]])
                        env_trajectories[env_idx].append(eef_state)
                
                # Get actions using the separated method
                action_chunk = self._get_flow_action_for_eval(obs)
                
                # Step EVAL environments
                obs, reward_venv, terminated_venv, truncated_venv, _ = self.eval_venv.step(action_chunk)
                done_venv = terminated_venv | truncated_venv
                
                # Store per-timestep rewards like pretraining
                reward_trajs.append(reward_venv)
                
                for env_idx in range(self.eval_n_envs):
                    if not episode_done[env_idx]:
                        env_rewards[env_idx] += reward_venv[env_idx]
                        if done_venv[env_idx]:
                            episode_done[env_idx] = True
                step_count += 1
                print(f"Pretrained eval episode {episode}, step {step_count}", end='\r')
            # Store trajectories from all environments for this episode
            for env_idx, traj in enumerate(env_trajectories):
                if len(traj) > 0:
                    traj_array = np.array(traj)
                    trajectories.append(traj_array)

            # Calculate success rate using the same method as pretraining
            if len(reward_trajs) > 0:
                reward_trajs_episode = np.stack(reward_trajs, axis=-1)  # (n_envs, timesteps)
                max_rewards_per_env = np.max(reward_trajs_episode, axis=1)  # (n_envs,) - max reward per env
                reward_trajs_all.extend(max_rewards_per_env.tolist())
            else:
                # Fallback to summed rewards if no timestep data
                reward_trajs_all.extend(env_rewards.tolist())
                
            rewards.extend(env_rewards.tolist())
        
        # Use pretraining's success calculation method
        mean_reward = np.mean(rewards)
        pretraining_style_success_rate = np.mean(np.array(reward_trajs_all) >= 1.0)
        original_success_rate = np.mean(np.array(rewards) >= 1.0)
        
        # log.info(f"Success rate comparison - Original method: {original_success_rate:.3f}, Pretraining method: {pretraining_style_success_rate:.3f}")
        success_rate = pretraining_style_success_rate
        
        log.info(f"Pretrained flow policy: reward={mean_reward:.3f}, success_rate={success_rate:.3f} "
                f"({len(trajectories)} trajectories)")
        
        # Calculate std reward
        std_reward = np.std(rewards) if len(rewards) > 0 else 0.0
        
        # Log pretrained evaluation results to CSV (step=0 since it's before training)
        self._log_eval_to_csv(
            step=0,
            eval_type='pretrained',
            mean_reward=mean_reward,
            std_reward=std_reward,
            success_rate=success_rate,
            mean_length=None,  # Not tracking length for pretrained eval
            num_episodes=len(trajectories)
        )
        
        return trajectories, mean_reward, success_rate
    
    def extract_expert_trajectories(self, num_trajectories: int = 160) -> List[np.ndarray]:
        """
        Extract expert trajectories from raw npz data, subsampled by horizon_steps.
        
        Args:
            num_trajectories: Number of expert trajectories to sample
            
        Returns:
            List of expert trajectory arrays with only robot EEF positions, subsampled
        """
        if not hasattr(self.cfg, 'expert_dataset') or 'dataset_path' not in self.cfg.expert_dataset:
            log.warning("No expert dataset path available for trajectory extraction")
            return []
        
        # Load raw npz data
        dataset_path = self.cfg.expert_dataset.dataset_path
        dataset = np.load(dataset_path, allow_pickle=False)
        states = dataset["states"]  # (total_steps, obs_dim)
        traj_lengths = dataset["traj_lengths"]
        
        expert_trajectories = []
        
        # Process each trajectory individually
        cumulative_length = 0
        for traj_idx in range(min(num_trajectories, len(traj_lengths))):
            traj_length = traj_lengths[traj_idx]
            traj_start = cumulative_length
            traj_end = cumulative_length + traj_length
            
            # Extract full trajectory states
            traj_states = states[traj_start:traj_end]  # (traj_length, obs_dim)
            
            # Subsample by horizon_steps (take states[0], states[8], states[16], ...)
            subsampled_indices = np.arange(0, traj_length, self.horizon_steps)
            subsampled_states = traj_states[subsampled_indices]  # (subsample_length, obs_dim)
            
            # Extract robot EEF positions: robot0_eef_pos [0:3] and robot1_eef_pos [9:12]
            robot0_eef = subsampled_states[:, :3]     # (subsample_length, 3)
            robot1_eef = subsampled_states[:, 9:12]   # (subsample_length, 3)
            eef_trajectory = np.concatenate([robot0_eef, robot1_eef], axis=1)  # (subsample_length, 6)
            
            expert_trajectories.append(eef_trajectory)
            cumulative_length += traj_length
        
        log.info(f"Extracted {len(expert_trajectories)} expert trajectories (subsampled by horizon_steps={self.horizon_steps}) for visualization")
        return expert_trajectories
    
    def create_trajectory_visualization(self, step: int, current_trajectories: List[np.ndarray]):
        """
        Create trajectory visualizations comparing current policy with pretrained baseline and expert data.
        
        Args:
            step: Current training step
            current_trajectories: Trajectories from current policy evaluation
        """
        if not self.visualize_trajectories:
            return
        
        log.info(f"Creating trajectory visualization at step {step}")
        
        # Use cached expert trajectories
        expert_trajectories = self.trajectory_data.get("expert", [])
        
        # Create pretrained vs finetuned comparison if available
        if "pretrained" in self.trajectory_data:
            pretrained_trajectories = self.trajectory_data["pretrained"]
            
            # Create pretrained vs finetuned comparison plots
            pretrained_dict = {
                "Pretrained": pretrained_trajectories,
                f"Step {step}": current_trajectories
            }
            
            pretrained_save_path = os.path.join(self.render_dir, f"pretrained_vs_finetuned_step_{step}.html")
            robot0_pretrained_fig, robot1_pretrained_fig = self.trajectory_visualizer.create_comparison_plot(
                trajectories_dict=pretrained_dict,
                title=f"Pretrained vs Finetuned: Step {step}",
                save_path=pretrained_save_path
            )
        
        # Create expert vs finetuned comparison plots
        expert_dict = {
            "Expert": expert_trajectories,
            f"Step {step}": current_trajectories
        }
        
        expert_save_path = os.path.join(self.render_dir, f"expert_vs_finetuned_step_{step}.html")
        robot0_expert_fig, robot1_expert_fig = self.trajectory_visualizer.create_comparison_plot(
            trajectories_dict=expert_dict,
            title=f"Expert vs Finetuned: Step {step}",
            save_path=expert_save_path
        )
        
        # Log to wandb
        if self.use_wandb:
            wandb_data = {}
            
            # Log pretrained vs finetuned comparison if available
            if "pretrained" in self.trajectory_data:
                robot0_pretrained_path = pretrained_save_path.replace('.html', '_robot0.html')
                robot1_pretrained_path = pretrained_save_path.replace('.html', '_robot1.html')
                
                wandb_data.update({
                    "robot0_pretrained_vs_finetuned_comparison": wandb.Html(open(robot0_pretrained_path).read()),
                    "robot1_pretrained_vs_finetuned_comparison": wandb.Html(open(robot1_pretrained_path).read())
                })
                
                # Also save as images for easier viewing
                try:
                    import plotly.io as pio
                    robot0_pretrained_img = robot0_pretrained_path.replace('.html', '.png')
                    robot1_pretrained_img = robot1_pretrained_path.replace('.html', '.png')
                    
                    pio.write_image(robot0_pretrained_fig, robot0_pretrained_img)
                    pio.write_image(robot1_pretrained_fig, robot1_pretrained_img)
                    
                    wandb_data.update({
                        "robot0_pretrained_vs_finetuned_image": wandb.Image(robot0_pretrained_img),
                        "robot1_pretrained_vs_finetuned_image": wandb.Image(robot1_pretrained_img)
                    })
                except Exception as e:
                    log.warning(f"Could not save pretrained comparison images: {e}")
            
            # Log expert vs finetuned comparison
            robot0_expert_path = expert_save_path.replace('.html', '_robot0.html')
            robot1_expert_path = expert_save_path.replace('.html', '_robot1.html')
            
            wandb_data.update({
                "robot0_expert_vs_finetuned_comparison": wandb.Html(open(robot0_expert_path).read()),
                "robot1_expert_vs_finetuned_comparison": wandb.Html(open(robot1_expert_path).read())
            })
            
            # Also save expert comparison as images
            try:
                import plotly.io as pio
                robot0_expert_img = robot0_expert_path.replace('.html', '.png')
                robot1_expert_img = robot1_expert_path.replace('.html', '.png')
                
                pio.write_image(robot0_expert_fig, robot0_expert_img)
                pio.write_image(robot1_expert_fig, robot1_expert_img)
                
                wandb_data.update({
                    "robot0_expert_vs_finetuned_image": wandb.Image(robot0_expert_img),
                    "robot1_expert_vs_finetuned_image": wandb.Image(robot1_expert_img)
                })
            except Exception as e:
                log.warning(f"Could not save expert comparison images: {e}")
            
            # Log all visualization data at once
            wandb.log(wandb_data, step=step*self.horizon_steps*self.n_envs)
        
        log.info(f"Saved trajectory visualizations for step {step}")

    def run(self):
        """
        Main training loop - proper off-policy RL (SAC-style) with env step + network update.
        """
        log.info("Starting distilled flow RL training")
        self.model.train()
        self.model.pretrained_flow_policy.eval()
        
        # Evaluate pretrained flow policy before any training (if evaluation is enabled)
        if self.cfg.get('run_eval', False) and hasattr(self, 'eval_venv'):
            pretrained_trajectories, pretrained_reward, pretrained_success = self.evaluate_pretrained_flow_policy()
        else:
            pretrained_trajectories, pretrained_reward, pretrained_success = [], 0.0, 0.0
            log.info("Skipping pretrained evaluation (run_eval=False or eval_venv not created)")

        if self.visualize_trajectories and hasattr(self.model, 'pretrained_flow_policy'):
            # Only store trajectories if evaluation was actually run
            if len(pretrained_trajectories) > 0:
                self.trajectory_data["pretrained"] = pretrained_trajectories  # Store ALL pretrained trajectories
                log.info(f"Stored {len(pretrained_trajectories)} pretrained trajectories for comparison")
            
            # Extract and cache expert trajectories for comparison (do this only once)
            expert_trajectories = self.extract_expert_trajectories(num_trajectories=self.max_n_episodes)
            self.trajectory_data["expert"] = expert_trajectories
            log.info(f"Stored {len(expert_trajectories)} expert trajectories for comparison")
            
        # Start training loop
        timer = Timer()
        # Reset step counter for online RL (warmstart steps don't count towards online training)
        total_steps = 0
        self.current_training_step = 0  # Initialize training step tracker
        
        # Reset environments initially
        prev_obs_venv = self.reset_env_all()
        episode_rewards = np.zeros(self.n_envs)
        episode_lengths = np.zeros(self.n_envs)
        episode_step_rewards = [[] for _ in range(self.n_envs)]  # Track individual step rewards
        completed_episodes_rewards = []  # Store completed episode rewards
        completed_episodes_lengths = []  # Store completed episode lengths
        
        while total_steps < self.num_train_steps:
            # Run evaluation and visualization (if evaluation is enabled)
            if self.cfg.get('run_eval', False) and hasattr(self, 'eval_venv') and total_steps % self.eval_freq == 0:
                # Evaluate and collect trajectories if visualization is enabled
                if self.visualize_trajectories and total_steps % self.visualization_freq == 0:
                    eval_reward, eval_success, eval_trajectories = self.evaluate(total_steps, collect_trajectories=True)
                    self.create_trajectory_visualization(total_steps, eval_trajectories)
                else:
                    self.evaluate(total_steps)
                
            # Get chunked actions and noise using online exploration strategy
            action_venv, noise_venv = self.get_action(prev_obs_venv, strategy=self.online_explore_strategy)
            
            # Extract action chunk for environment step
            action_chunk = action_venv[:, : self.act_steps]  # (n_envs, act_steps, action_dim)
            
            # # Step environment
            obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = self.venv.step(action_chunk.cpu().numpy())
            done_venv = terminated_venv | truncated_venv
            
            # Convert rewards and done flags to tensors
            reward_venv_tensor = torch.tensor(reward_venv, dtype=torch.float32, device=self.device)
            done_venv_tensor = torch.tensor(done_venv, dtype=torch.bool, device=self.device)
            
            total_reward = reward_venv_tensor.unsqueeze(-1)  # (n_envs, 1)
            if total_steps % self.log_freq == 0:
                self.last_intrinsic_reward = 0.0
                self.last_env_reward = reward_venv_tensor.mean().item()
            
            # Add transitions to replay buffer (with info for full trajectory data)
            self.collect_transition(
                prev_obs_venv, action_chunk, noise_venv, total_reward, done_venv_tensor, obs_venv, info_venv
            )
            
            # Handle environment resets and collect completed episodes
            for env_idx in range(self.n_envs):
                # Track step rewards for better episode boundary handling
                episode_step_rewards[env_idx].append(reward_venv[env_idx])
                
                if done_venv[env_idx]:
                    # Calculate true episode reward from tracked steps
                    true_episode_reward = sum(episode_step_rewards[env_idx])
                    true_episode_length = len(episode_step_rewards[env_idx])
                    completed_episodes_rewards.append(true_episode_reward)
                    completed_episodes_lengths.append(true_episode_length)
                    
                    # Reset episode tracking for this env
                    episode_rewards[env_idx] = 0
                    episode_lengths[env_idx] = 0
                    episode_step_rewards[env_idx] = []
                else:
                    # Update running statistics
                    episode_rewards[env_idx] += reward_venv[env_idx]
                    episode_lengths[env_idx] += 1
            
            # Log training metrics periodically (averaged across completed episodes)
            if total_steps % self.log_freq == 0 and len(completed_episodes_rewards) > 0:
                avg_train_reward = np.mean(completed_episodes_rewards)
                max_train_reward = np.max(completed_episodes_rewards)
                avg_train_length = np.mean(completed_episodes_lengths)
                
                # Compute training success rate (assuming binary rewards: 0=failure, >=1=success)
                train_success_rate = np.mean(np.array(completed_episodes_rewards) >= 1.0)
                
                if self.use_wandb:
                    wandb.log({
                        'train/episode_reward': avg_train_reward,
                        'train/max_episode_reward': max_train_reward,
                        'train/episode_length': avg_train_length,
                        'train/success_rate': train_success_rate,
                        'buffer/size': len(self.replay_buffer),
                        'buffer/num_online_episodes': self.replay_buffer.num_episodes,
                    }, step=total_steps*self.horizon_steps*self.n_envs)
                
                # Clear completed episodes (or keep a sliding window)
                completed_episodes_rewards = completed_episodes_rewards[-100:]  # Keep last 100
                completed_episodes_lengths = completed_episodes_lengths[-100:]  # Keep last 100
            
            # Update networks
            # Skip updates during initial exploration phase
            if (self.replay_buffer.get_total_transitions() >= self.batch_size and 
                total_steps % self.update_freq == 0):
                
                # Accumulate metrics to log once at the end
                accumulated_wandb_dict = {}
                
                for _ in range(self.gradient_steps):
                    loss_dict, current_expert_ratio = self.update_networks(training_step=total_steps)
                    if loss_dict is not None:
                        # Accumulate training metrics (only from last gradient step)
                        if _ == self.gradient_steps - 1 and total_steps % self.log_freq == 0:
                            if self.use_wandb:
                                wandb_dict = {
                                    'train/actor_loss': loss_dict['actor_total'].item(),
                                    'train/actor_bc_loss': loss_dict['actor_bc_loss'].item(),
                                    'train/actor_q_loss': loss_dict['actor_q_loss'].item(),
                                    'train/critic_loss': loss_dict['critic_loss'].item(), 
                                    # Q-filtering metrics
                                    'train/q_advantage_mean': loss_dict['q_advantage_mean'].item(),
                                    'train/better_than_expert_percentage': loss_dict['better_than_expert_percentage'].item(),
                                    'train/pretrained_q_mean': loss_dict['pretrained_q_mean'].item(),
                                    'train/current_q_mean': loss_dict['current_q_mean'].item(),
                                    'train/q_filtering_active': loss_dict['q_filtering_active'].item(),
                                    'train/env_reward_mean': getattr(self, 'last_env_reward', 0.0),
                                    # Gradient norm metrics
                                    'train/actor_grad_norm': loss_dict['actor_grad_norm'],
                                    'train/critic_grad_norm': loss_dict['critic_grad_norm'],
                                    # Delayed update metrics
                                    # 'train/actor_updated': loss_dict['actor_updated'],
                                    # 'train/target_updated': loss_dict['target_updated'],
                                }
                                
                                # Add Q-value overestimation metrics if available
                                for key, value in loss_dict.items():
                                    if key.startswith('q_analysis/'):
                                        wandb_dict[key] = value
                                
                                # Add adaptive expert ratio if enabled
                                if self.use_adaptive_expert_ratio:
                                    wandb_dict['train/expert_ratio'] = current_expert_ratio
                                
                                # Store for later logging
                                accumulated_wandb_dict.update(wandb_dict)
            
                # Log all accumulated metrics at once with the current step
                # This ensures we only call wandb.log once per environment step
                if self.use_wandb and accumulated_wandb_dict:
                    wandb.log(accumulated_wandb_dict, step=total_steps*self.horizon_steps*self.n_envs)
            
            # Update for next step
            prev_obs_venv = obs_venv
            total_steps += 1
            self.current_training_step = total_steps  # Update instance variable
            
            # Save model and log progress periodically
            if total_steps % self.save_freq == 0 and total_steps > 0:
                self.save_model(total_steps)
                log.info(f"Step {total_steps}/{self.num_train_steps}: "
                        f"buffer_size={len(self.replay_buffer)}")
            
        # Final evaluation and save
        self.evaluate(total_steps)
        # self.save_model(total_steps)
        log.info("Training completed")
    
    def save_model(self, step=None):
        """Save the model and training state."""
        if step is None:
            step = 0
        save_path = os.path.join(self.logdir, f"model_step_{step}.pth")
        
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'step': step,
            'config': self.cfg,
        }

        if self.actor_scheduler is not None:
            save_dict['actor_scheduler_state_dict'] = self.actor_scheduler.state_dict()
        if self.critic_scheduler is not None:
            save_dict['critic_scheduler_state_dict'] = self.critic_scheduler.state_dict()
        
        torch.save(save_dict, save_path)
        log.info(f"Model saved to {save_path}")
    
    def load(self, step):
        """Load the model and training state."""
        load_path = os.path.join(self.logdir, f"model_step_{step}.pth")
        
        if not os.path.exists(load_path):
            log.warning(f"Model file {load_path} not found")
            return
        
        checkpoint = torch.load(load_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        if 'actor_scheduler_state_dict' in checkpoint and self.actor_scheduler is not None:
            self.actor_scheduler.load_state_dict(checkpoint['actor_scheduler_state_dict'])
        if 'critic_scheduler_state_dict' in checkpoint and self.critic_scheduler is not None:
            self.critic_scheduler.load_state_dict(checkpoint['critic_scheduler_state_dict'])
        
        # Load replay buffer
        self.replay_buffer = checkpoint['replay_buffer']
        
        loaded_step = checkpoint.get('step', 0)
        log.info(f"Model loaded from {load_path} at step {loaded_step}")
    
    def _setup_upsampling_wrapper(self, cfg):
        """
        Helper method to set up wrapper configuration for upsampling.
        Replaces multi_step with multi_step_full and fixes config interpolations.
        """
        if not (hasattr(cfg.env, 'wrappers') and 'multi_step' in cfg.env.wrappers):
            return cfg
        
        from omegaconf import OmegaConf
        
        # Preserve resolved timestamp values before modification
        resolved_values = {}
        for key in ['logdir', 'name']:
            if hasattr(cfg, key):
                try:
                    resolved_values[key] = OmegaConf.select(cfg, key)
                except:
                    pass
        
        # Also preserve wandb run name if it exists
        if hasattr(cfg, 'wandb') and hasattr(cfg.wandb, 'run'):
            try:
                resolved_values['wandb_run'] = OmegaConf.select(cfg, 'wandb.run')
            except:
                pass
        
        # Convert to unresolved container to handle interpolations
        cfg_dict = OmegaConf.to_container(cfg, resolve=False, enum_to_str=True)
        
        # Replace wrapper name in config
        if 'env' in cfg_dict and 'wrappers' in cfg_dict['env'] and 'multi_step' in cfg_dict['env']['wrappers']:
            cfg_dict['env']['wrappers']['multi_step_full'] = cfg_dict['env']['wrappers'].pop('multi_step')
            
            # Fix interpolation references throughout config
            self._fix_wrapper_interpolations(cfg_dict, '${env.wrappers.multi_step.', '${env.wrappers.multi_step_full.')
        
        # Create new config
        new_cfg = OmegaConf.create(cfg_dict)
        
        # Restore resolved timestamp values to prevent re-evaluation
        for key, value in resolved_values.items():
            if key == 'wandb_run':
                if hasattr(new_cfg, 'wandb'):
                    new_cfg.wandb.run = value
            else:
                setattr(new_cfg, key, value)
        
        return new_cfg
    
    def _fix_wrapper_interpolations(self, obj, old_ref, new_ref):
        """Recursively fix interpolation references in config object."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, str) and old_ref in value:
                    obj[key] = value.replace(old_ref, new_ref)
                elif isinstance(value, (dict, list)):
                    self._fix_wrapper_interpolations(value, old_ref, new_ref)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                if isinstance(item, str) and old_ref in item:
                    obj[i] = item.replace(old_ref, new_ref)
                elif isinstance(item, (dict, list)):
                    self._fix_wrapper_interpolations(item, old_ref, new_ref)
        


