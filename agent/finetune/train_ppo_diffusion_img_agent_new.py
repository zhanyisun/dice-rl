"""
PPO fine-tuning with image observations.
Extracts visual features once using frozen encoder and works with augmented state.
"""

import os
import pickle
import einops
import numpy as np
import torch
import logging
import wandb
import math
from omegaconf import OmegaConf

log = logging.getLogger(__name__)
from util.timer import Timer
from agent.finetune.train_ppo_diffusion_agent import TrainPPODiffusionAgent


class TrainPPODiffusionImgAgent(TrainPPODiffusionAgent):
    """
    PPO agent for image-based observations.
    Key approach:
    1. Extract visual features from RGB using frozen pretrained encoder
    2. Concatenate visual features with state observations
    3. Store and work with augmented features throughout (never raw RGB during RL)
    """
    
    def __init__(self, cfg):
        # Store dimensions from config
        self.original_obs_dim = cfg.original_obs_dim  # Low-dim state
        self.visual_feature_dim = cfg.visual_feature_dim  # Visual features
        self.obs_dim = cfg.obs_dim  # Should be visual_feature_dim + original_obs_dim
        
        # Verify dimensions match
        assert self.obs_dim == self.visual_feature_dim + self.original_obs_dim, \
            f"obs_dim mismatch: {self.obs_dim} != {self.visual_feature_dim} + {self.original_obs_dim}"
        
        # Call parent init with correct obs_dim
        super().__init__(cfg)
        
        # Freeze visual encoder components and set to eval mode
        # Assert that actor_ft exists (should be created by VPGDiffusion)
        assert hasattr(self.model, 'actor_ft'), "Model must have actor_ft for fine-tuning"
        
        # Put pretrained actor always in eval mode (it's already frozen by VPGDiffusion)
        self.model.actor.eval()
        
        # Assert visual encoder components exist in actor_ft
        assert hasattr(self.model.actor_ft, 'encoders'), "actor_ft must have encoders for visual feature extraction"
        assert hasattr(self.model.actor_ft, 'visual_proj'), "actor_ft must have visual_proj for visual feature projection"
        
        # Freeze ResNet encoders in actor_ft and set to eval
        for encoder in self.model.actor_ft.encoders:
            encoder.eval()
            for param in encoder.parameters():
                param.requires_grad = False
        log.info(f"Froze {len(self.model.actor_ft.encoders)} visual encoders")
        
        # Freeze visual projection layer in actor_ft and set to eval
        self.model.actor_ft.visual_proj.eval()
        for param in self.model.actor_ft.visual_proj.parameters():
            param.requires_grad = False
        log.info("Froze visual projection layer")
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.model.actor_ft.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.actor_ft.parameters())
        log.info(f"Trainable parameters in actor_ft: {trainable_params}/{total_params}")
        
        log.info(f"PPO Image Agent initialized:")
        log.info(f"  Original obs_dim: {self.original_obs_dim}")
        log.info(f"  Visual feature dim: {self.visual_feature_dim}")
        log.info(f"  Augmented obs_dim: {self.obs_dim}")
        
    def _extract_and_merge_features(self, obs_venv):
        """
        Extract visual features from RGB and merge with state observations.
        This is done once per timestep, and the augmented features are used throughout.
        
        Args:
            obs_venv: Dict with 'rgb' and 'state' keys from environment
                state: (n_envs, cond_steps, state_dim)
                rgb: (n_envs, cond_steps, C*num_cameras, H, W)
                
        Returns:
            augmented_state: (n_envs, cond_steps, obs_dim) tensor
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

        # Extract visual features using the frozen pretrained encoder
        # The actor contains the visual encoder from pretraining
        with torch.no_grad():
            # Use the pretrained actor (self.model.actor) to extract features
            # Note: actor is the frozen pretrained model, actor_ft is the fine-tuned one
            visual_features = self.model.actor.extract_visual_features(cond)
            # visual_features shape: (n_envs, visual_feature_dim)
        
        # Merge visual features with original state
        # Repeat visual features for each cond_step
        if state.shape[1] > 1:  # Has cond_steps dimension
            visual_features = visual_features.unsqueeze(1).repeat(1, state.shape[1], 1)
        else:
            visual_features = visual_features.unsqueeze(1)
        
        augmented_state = torch.cat([state, visual_features], dim=-1)

        return augmented_state  # Return as tensor
    
    def evaluate(self, cnt_train_step: int):
        """
        Evaluate the agent using augmented features.
        """
        self.model.eval()
        
        eval_episode_rewards = []
        eval_episode_lengths = []
        
        for episode in range(self.num_eval_episodes):
            # Prepare options with fixed seeds
            options_venv = [{} for _ in range(self.eval_n_envs)]
            for env_idx in range(self.eval_n_envs):
                seed_idx = episode * self.eval_n_envs + env_idx
                options_venv[env_idx]["seed"] = self.eval_seeds[seed_idx]
            
            # Add video paths for first episode
            if self.render_video and episode == 0:
                for env_ind in range(min(self.n_render, self.eval_n_envs)):
                    options_venv[env_ind]["video_path"] = os.path.join(
                        self.render_dir, f"step-{cnt_train_step}_eval_episode-{episode}_env-{env_ind}.mp4"
                    )
            
            # Reset environments
            obs = self.reset_env_all(options_venv=options_venv, eval_flag=True)
            episode_rewards = np.zeros(self.eval_n_envs)
            episode_lengths = np.zeros(self.eval_n_envs)
            episode_done = np.zeros(self.eval_n_envs, dtype=bool)
            
            step = 0
            max_eval_steps = math.ceil(self.max_episode_steps / self.act_steps)
            
            while step < max_eval_steps:
                with torch.no_grad():
                    # Extract visual features and merge with state
                    augmented_state = self._extract_and_merge_features(obs)
                    
                    # Get actions using pre-extracted features
                    samples = self.model.forward_from_features(
                        features=augmented_state,
                        deterministic=True,
                        return_chain=False,
                        return_noise=False,
                    )
                    action_venv = samples.trajectories.cpu().numpy()
                    action_chunk = action_venv[:, : self.act_steps]
                
                # Step environments
                obs, reward_venv, terminated_venv, truncated_venv, info_venv = self.eval_venv.step(action_chunk)
                step += 1
                done_venv = terminated_venv | truncated_venv
                
                # Update episode statistics
                for env_idx in range(self.eval_n_envs):
                    if not episode_done[env_idx]:
                        episode_rewards[env_idx] += reward_venv[env_idx]
                        episode_lengths[env_idx] += self.act_steps
                        
                        if done_venv[env_idx] or episode_lengths[env_idx] >= self.max_episode_steps:
                            episode_done[env_idx] = True
            
            eval_episode_rewards.extend(episode_rewards.tolist())
            eval_episode_lengths.extend(episode_lengths.tolist())
        
        # Calculate metrics
        mean_episode_reward = np.mean(eval_episode_rewards)
        std_episode_reward = np.std(eval_episode_rewards)
        mean_episode_length = np.mean(eval_episode_lengths)
        success_rate = np.mean(np.array(eval_episode_rewards) >= self.best_reward_threshold_for_success)
        
        log.info(f"Eval at step {cnt_train_step}: "
                f"reward={mean_episode_reward:.3f}�{std_episode_reward:.3f}, "
                f"length={mean_episode_length:.1f}, "
                f"success_rate={success_rate:.3f} "
                f"({len(eval_episode_rewards)} episodes)")
        
        if self.use_wandb:
            wandb.log({
                'eval/episode_reward': mean_episode_reward,
                'eval/episode_reward_std': std_episode_reward,
                'eval/episode_length': mean_episode_length,
                'eval/success_rate': success_rate,
                'eval/num_episodes': len(eval_episode_rewards),
            }, step=cnt_train_step, commit=False)
        
        self._log_eval_to_csv(
            step=cnt_train_step,
            eval_type='finetuned',
            mean_reward=mean_episode_reward,
            std_reward=std_episode_reward,
            success_rate=success_rate,
            mean_length=mean_episode_length,
            num_episodes=len(eval_episode_rewards)
        )
        
        self.model.train()
        # Keep pretrained actor and visual encoders in eval mode even after calling train()
        self.model.actor.eval()  # Pretrained model always in eval
        for encoder in self.model.actor_ft.encoders:
            encoder.eval()
        self.model.actor_ft.visual_proj.eval()
        return mean_episode_reward, success_rate
    
    def run(self):
        """
        Main training loop with visual feature extraction.
        """
        timer = Timer()
        run_results = []
        cnt_train_step = 0
        done_venv = np.zeros((1, self.n_envs))
        
        while self.itr < self.n_train_itr:
            # Prepare video paths
            options_venv = [{} for _ in range(self.n_envs)]
            if self.itr % self.render_freq == 0 and self.render_video:
                for env_ind in range(self.n_render):
                    options_venv[env_ind]["video_path"] = os.path.join(
                        self.render_dir, f"itr-{self.itr}_trial-{env_ind}.mp4"
                    )
            
            self.model.train()
            # Keep pretrained actor and visual encoders in eval mode even after calling train()
            self.model.actor.eval()  # Pretrained model always in eval
            for encoder in self.model.actor_ft.encoders:
                encoder.eval()
            self.model.actor_ft.visual_proj.eval()
            
            # Reset environments if needed
            firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))
            if self.reset_at_iteration:
                prev_obs_venv = self.reset_env_all(options_venv=options_venv)
                firsts_trajs[0] = 1
            else:
                firsts_trajs[0] = done_venv
            
            # Initialize trajectory holders
            # IMPORTANT: We store augmented features in obs_trajs["state"], not raw observations
            obs_trajs = {
                "state": np.zeros(
                    (self.n_steps, self.n_envs, self.n_cond_step, self.obs_dim)
                )
            }
            chains_trajs = np.zeros(
                (
                    self.n_steps,
                    self.n_envs,
                    self.model.ft_denoising_steps + 1,
                    self.horizon_steps,
                    self.action_dim,
                )
            )
            noise_trajs = np.zeros(
                (
                    self.n_steps,
                    self.n_envs,
                    self.model.ft_denoising_steps + 1,
                    self.horizon_steps,
                    self.action_dim,
                )
            )
            terminated_trajs = np.zeros((self.n_steps, self.n_envs))
            reward_trajs = np.zeros((self.n_steps, self.n_envs))
            
            if self.save_full_observations:
                obs_full_trajs = np.empty((0, self.n_envs, self.obs_dim))
                # Extract features for initial observation
                augmented_initial = self._extract_and_merge_features(prev_obs_venv)
                obs_full_trajs = np.vstack(
                    (obs_full_trajs, augmented_initial[:, -1].cpu().numpy()[None])
                )
            
            # Collect trajectories
            for step in range(self.n_steps):
                # Run evaluation periodically using separate eval environments
                if self.run_eval and hasattr(self, 'eval_venv') and cnt_train_step % (self.eval_freq * self.n_envs * self.act_steps) == 0:
                    self.evaluate(cnt_train_step)

                if step % 10 == 0:
                    print(f"Processed step {step} of {self.n_steps}")
                
                with torch.no_grad():
                    # Extract visual features and merge with state
                    augmented_state = self._extract_and_merge_features(prev_obs_venv)
                    
                    # Store augmented features in trajectory buffer
                    obs_trajs["state"][step] = augmented_state.cpu().numpy()
                    
                    # Get actions using augmented features
                    cond = {"state": augmented_state}
                    samples, noises = self.model.forward_from_features(
                        features=augmented_state,
                        deterministic=False,
                        return_chain=True,
                        return_noise=True,
                    )
                    output_venv = samples.trajectories.cpu().numpy()
                    chains_venv = samples.chains.cpu().numpy()
                
                action_venv = output_venv[:, : self.act_steps]
                
                # Step environments
                obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = (
                    self.venv.step(action_venv)
                )
                done_venv = terminated_venv | truncated_venv
                
                if self.save_full_observations:
                    # Extract features for full observations
                    augmented_full = self._extract_and_merge_features({
                        "state": np.array([info["full_obs"]["state"] for info in info_venv]),
                        "rgb": np.array([info["full_obs"]["rgb"] for info in info_venv])
                    })
                    obs_full_trajs = np.vstack(
                        (obs_full_trajs, augmented_full.transpose(1, 0, 2).cpu().numpy())
                    )
                
                chains_trajs[step] = chains_venv
                reward_trajs[step] = reward_venv
                terminated_trajs[step] = terminated_venv
                firsts_trajs[step + 1] = done_venv
                
                prev_obs_venv = obs_venv
                cnt_train_step += self.n_envs * self.act_steps
            
            # Calculate episode statistics
            episodes_start_end = []
            for env_ind in range(self.n_envs):
                env_steps = np.where(firsts_trajs[:, env_ind] == 1)[0]
                for i in range(len(env_steps) - 1):
                    start = env_steps[i]
                    end = env_steps[i + 1]
                    if end - start > 1:
                        episodes_start_end.append((env_ind, start, end - 1))
            
            if len(episodes_start_end) > 0:
                reward_trajs_split = [
                    reward_trajs[start : end + 1, env_ind]
                    for env_ind, start, end in episodes_start_end
                ]
                num_episode_finished = len(reward_trajs_split)
                episode_reward = np.array(
                    [np.sum(reward_traj) for reward_traj in reward_trajs_split]
                )
                if self.furniture_sparse_reward:
                    episode_best_reward = episode_reward
                else:
                    episode_best_reward = np.array(
                        [np.max(reward_traj) / self.act_steps for reward_traj in reward_trajs_split]
                    )
                avg_episode_reward = np.mean(episode_reward)
                avg_best_reward = np.mean(episode_best_reward)
                success_rate = np.mean(episode_best_reward >= self.best_reward_threshold_for_success)
            else:
                episode_reward = np.array([])
                num_episode_finished = 0
                avg_episode_reward = 0
                avg_best_reward = 0
                success_rate = 0
                log.info("[WARNING] No episode completed within the iteration!")
            
            # Update models with augmented features
            with torch.no_grad():
                # Convert augmented features to tensor
                obs_trajs["state"] = torch.from_numpy(obs_trajs["state"]).float().to(self.device)
                
                # Calculate values and logprobs with augmented features
                num_split = math.ceil(self.n_envs * self.n_steps / self.logprob_batch_size)
                obs_ts = [{} for _ in range(num_split)]
                obs_k = einops.rearrange(obs_trajs["state"], "s e ... -> (s e) ...")
                obs_ts_k = torch.split(obs_k, self.logprob_batch_size, dim=0)
                for i, obs_t in enumerate(obs_ts_k):
                    obs_ts[i]["state"] = obs_t
                
                values_trajs = np.empty((0, self.n_envs))
                for obs in obs_ts:
                    # Critic uses augmented features
                    values = self.model.critic(obs).cpu().numpy().flatten()
                    values_trajs = np.vstack((values_trajs, values.reshape(-1, self.n_envs)))
                
                chains_t = einops.rearrange(
                    torch.from_numpy(chains_trajs).float().to(self.device),
                    "s e t h d -> (s e) t h d",
                )
                chains_ts = torch.split(chains_t, self.logprob_batch_size, dim=0)
                logprobs_trajs = np.empty(
                    (0, self.model.ft_denoising_steps, self.horizon_steps, self.action_dim)
                )
                for obs, chains in zip(obs_ts, chains_ts):
                    # Get logprobs using augmented features
                    logprobs = self.model.get_logprobs_from_features(obs, chains).cpu().numpy()
                    logprobs_trajs = np.vstack(
                        (logprobs_trajs, logprobs.reshape(-1, *logprobs_trajs.shape[1:]))
                    )
                
                # Normalize rewards if needed
                if self.reward_scale_running:
                    reward_trajs_transpose = self.running_reward_scaler(
                        reward=reward_trajs.T, first=firsts_trajs[:-1].T
                    )
                    reward_trajs = reward_trajs_transpose.T
                
                # GAE calculation - need to extract features for final observation
                final_augmented = self._extract_and_merge_features(obs_venv)
                obs_venv_ts = {"state": final_augmented}
                
                advantages_trajs = np.zeros_like(reward_trajs)
                lastgaelam = 0
                for t in reversed(range(self.n_steps)):
                    if t == self.n_steps - 1:
                        nextvalues = self.model.critic(obs_venv_ts).reshape(1, -1).cpu().numpy()
                    else:
                        nextvalues = values_trajs[t + 1]
                    nonterminal = 1.0 - terminated_trajs[t]
                    delta = (
                        reward_trajs[t] * self.reward_scale_const
                        + self.gamma * nextvalues * nonterminal
                        - values_trajs[t]
                    )
                    advantages_trajs[t] = lastgaelam = (
                        delta + self.gamma * self.gae_lambda * nonterminal * lastgaelam
                    )
                returns_trajs = advantages_trajs + values_trajs
            
            # Prepare data for PPO updates
            obs_k = {"state": einops.rearrange(obs_trajs["state"], "s e ... -> (s e) ...")}
            chains_k = einops.rearrange(
                torch.tensor(chains_trajs).float().to(self.device),
                "s e t h d -> (s e) t h d",
            )
            returns_k = torch.tensor(returns_trajs).float().to(self.device).reshape(-1)
            values_k = torch.tensor(values_trajs).float().to(self.device).reshape(-1)
            advantages_k = torch.tensor(advantages_trajs).float().to(self.device).reshape(-1)
            logprobs_k = torch.tensor(logprobs_trajs).float().to(self.device)
            
            # PPO update epochs
            total_steps = self.n_steps * self.n_envs
            inds_k = np.arange(total_steps)
            clipfracs = []
            
            for update_epoch in range(self.update_epochs):
                flag_break = False
                np.random.shuffle(inds_k)
                num_batch = max(1, total_steps // self.batch_size)
                
                for batch in range(num_batch):
                    start = batch * self.batch_size
                    end = start + self.batch_size
                    inds_b = inds_k[start:end]
                    
                    # All observations are augmented features
                    obs_b = {"state": obs_k["state"][inds_b]}
                    chains_b = chains_k[inds_b]
                    returns_b = returns_k[inds_b]
                    values_b = values_k[inds_b]
                    advantages_b = advantages_k[inds_b]
                    logprobs_b = logprobs_k[inds_b]
                    
                    # Get PPO loss - model works with augmented features
                    (
                        pg_loss,
                        entropy_loss,
                        v_loss,
                        clipfrac,
                        approx_kl,
                        ratio,
                        bc_loss,
                        eta,
                    ) = self.model.loss_from_features(
                        obs_b,
                        chains_b,
                        returns_b,
                        values_b,
                        advantages_b,
                        logprobs_b,
                        use_bc_loss=self.use_bc_loss,
                        reward_horizon=self.reward_horizon,
                    )
                    loss = (
                        pg_loss
                        + entropy_loss * self.ent_coef
                        + v_loss * self.vf_coef
                        + bc_loss * self.bc_loss_coeff
                    )
                    clipfracs += [clipfrac]
                    
                    # Gradient updates
                    self.actor_optimizer.zero_grad()
                    self.critic_optimizer.zero_grad()
                    if self.learn_eta:
                        self.eta_optimizer.zero_grad()
                    loss.backward()
                    
                    if self.itr >= self.n_critic_warmup_itr:
                        if self.max_grad_norm is not None:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.actor_ft.parameters(), self.max_grad_norm
                            )
                        self.actor_optimizer.step()
                        if self.learn_eta and batch % self.eta_update_interval == 0:
                            self.eta_optimizer.step()
                    self.critic_optimizer.step()
                    
                    log.info(f"approx_kl: {approx_kl}, update_epoch: {update_epoch}, num_batch: {num_batch}")
                    
                    if self.target_kl is not None and approx_kl > self.target_kl:
                        flag_break = True
                        break
                
                if flag_break:
                    break
            
            # Calculate explained variance
            y_pred, y_true = values_k.cpu().numpy(), returns_k.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            
            # Plot trajectories if applicable
            if (
                self.itr % self.render_freq == 0
                and self.n_render > 0
                and self.traj_plotter is not None
            ):
                self.traj_plotter(
                    obs_full_trajs=obs_full_trajs,
                    n_render=self.n_render,
                    max_episode_steps=self.max_episode_steps,
                    render_dir=self.render_dir,
                    itr=self.itr,
                )
            
            # Update schedulers
            if self.itr >= self.n_critic_warmup_itr:
                self.actor_lr_scheduler.step()
                if self.learn_eta:
                    self.eta_lr_scheduler.step()
            self.critic_lr_scheduler.step()
            self.model.step()
            diffusion_min_sampling_std = self.model.get_min_sampling_denoising_std()
            
            # Save model
            if self.itr % self.save_model_freq == 0 or self.itr == self.n_train_itr - 1:
                self.save_model()
            
            # Log and save results
            run_results.append({"itr": self.itr, "step": cnt_train_step})
            if self.save_trajs:
                run_results[-1]["obs_full_trajs"] = obs_full_trajs if self.save_full_observations else None
                run_results[-1]["obs_trajs"] = obs_trajs
                run_results[-1]["chains_trajs"] = chains_trajs
                run_results[-1]["reward_trajs"] = reward_trajs
                
            if self.itr % self.log_freq == 0:
                time = timer()
                run_results[-1]["time"] = time
                log.info(
                    f"{self.itr}: step {cnt_train_step:8d} | loss {loss:8.4f} | "
                    f"pg loss {pg_loss:8.4f} | value loss {v_loss:8.4f} | "
                    f"bc loss {bc_loss:8.4f} | reward {avg_episode_reward:8.4f} | "
                    f"eta {eta:8.4f} | t:{time:8.4f}"
                )
                
                if self.use_wandb:
                    wandb.log(
                        {
                            "total env step": cnt_train_step,
                            "loss": loss,
                            "pg loss": pg_loss,
                            "value loss": v_loss,
                            "bc loss": bc_loss,
                            "eta": eta,
                            "approx kl": approx_kl,
                            "ratio": ratio,
                            "clipfrac": np.mean(clipfracs),
                            "explained variance": explained_var,
                            "avg episode reward - train": avg_episode_reward,
                            "num episode - train": num_episode_finished,
                            "diffusion - min sampling std": diffusion_min_sampling_std,
                            "actor lr": self.actor_optimizer.param_groups[0]["lr"],
                            "critic lr": self.critic_optimizer.param_groups[0]["lr"],
                        },
                        step=cnt_train_step,
                        commit=True,
                    )
                
                run_results[-1]["train_episode_reward"] = avg_episode_reward
                with open(self.result_path, "wb") as f:
                    pickle.dump(run_results, f)
                    
            self.itr += 1