"""
DPPO fine-tuning.

"""

import os
import pickle
import einops
import numpy as np
import torch
import logging
import wandb
import math
import csv

log = logging.getLogger(__name__)
from util.timer import Timer
from agent.finetune.train_ppo_agent import TrainPPOAgent
from util.scheduler import CosineAnnealingWarmupRestarts


class TrainPPODiffusionAgent(TrainPPOAgent):

    def __init__(self, cfg):
        super().__init__(cfg)

        # Reward horizon --- always set to act_steps for now
        self.reward_horizon = cfg.get("reward_horizon", self.act_steps)
        
        # Evaluation settings (matching train_distill_flow_agent.py)
        self.run_eval = cfg.get("run_eval", False)
        self.num_eval_episodes = cfg.get("num_eval_episodes", 5)  # Number of eval episodes
        self.eval_freq = cfg.train.get("eval_freq", 1000)  # Evaluate every N training steps
        
        # Fixed seeds for evaluation: num_eval_episodes * eval_n_envs total seeds
        if self.run_eval and hasattr(self, 'eval_n_envs'):
            self.eval_seeds = list(range(10000, 10000 + self.num_eval_episodes * self.eval_n_envs))
            log.info(f"Evaluation enabled with {self.num_eval_episodes} episodes x {self.eval_n_envs} envs = {self.num_eval_episodes * self.eval_n_envs} total episodes")
            
            # Initialize CSV file for evaluation results
            self.eval_csv_path = os.path.join(self.logdir, "evaluation_results.csv")
            self._init_eval_csv()

        # Eta - between DDIM (=0 for eval) and DDPM (=1 for training)
        self.learn_eta = self.model.learn_eta
        if self.learn_eta:
            self.eta_update_interval = cfg.train.eta_update_interval
            self.eta_optimizer = torch.optim.AdamW(
                self.model.eta.parameters(),
                lr=cfg.train.eta_lr,
                weight_decay=cfg.train.eta_weight_decay,
            )
            self.eta_lr_scheduler = CosineAnnealingWarmupRestarts(
                self.eta_optimizer,
                first_cycle_steps=cfg.train.eta_lr_scheduler.first_cycle_steps,
                cycle_mult=1.0,
                max_lr=cfg.train.eta_lr,
                min_lr=cfg.train.eta_lr_scheduler.min_lr,
                warmup_steps=cfg.train.eta_lr_scheduler.warmup_steps,
                gamma=1.0,
            )
    
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
                'mean_length': mean_length,
                'num_episodes': num_episodes
            })

    def evaluate(self, cnt_train_step: int):
        """
        Evaluate the agent using separate evaluation environments with fixed seeds.
        
        Args:
            cnt_train_step: Current training step for logging
        """
        self.model.eval()
        
        eval_episode_rewards = []
        eval_episode_lengths = []
        
        # Run num_eval_episodes sequential episodes, each using eval_n_envs in parallel
        for episode in range(self.num_eval_episodes):
            # Prepare options with fixed seeds for reproducible evaluation
            options_venv = [{} for _ in range(self.eval_n_envs)]
            for env_idx in range(self.eval_n_envs):
                # Use fixed seeds for evaluation - different seeds for each episode+env combination
                seed_idx = episode * self.eval_n_envs + env_idx
                options_venv[env_idx]["seed"] = self.eval_seeds[seed_idx]
            
            # Add video paths for first few environments
            if self.render_video and episode == 0:  # Only render first episode
                for env_ind in range(min(self.n_render, self.eval_n_envs)):
                    options_venv[env_ind]["video_path"] = os.path.join(
                        self.render_dir, f"step-{cnt_train_step}_eval_episode-{episode}_env-{env_ind}.mp4"
                    )
            
            # Reset all EVAL environments for this episode with fixed seeds
            obs = self.reset_env_all(options_venv=options_venv, eval_flag=True)
            episode_rewards = np.zeros(self.eval_n_envs)
            episode_lengths = np.zeros(self.eval_n_envs)
            episode_done = np.zeros(self.eval_n_envs, dtype=bool)
            
            step = 0
            # Run episode until all environments are done or max steps reached
            # Use ceiling division since we're executing act_steps actions at once
            max_eval_steps = math.ceil(self.max_episode_steps / self.act_steps)
            
            while step < max_eval_steps:
                # Get actions deterministically for evaluation
                with torch.no_grad():
                    cond = {
                        "state": torch.from_numpy(obs["state"])
                        .float()
                        .to(self.device)
                    }
                    samples = self.model(
                        cond=cond,
                        deterministic=True,  # Always deterministic for eval
                        return_chain=False,
                        return_noise=False,
                    )
                    action_venv = samples.trajectories.cpu().numpy()
                    action_chunk = action_venv[:, : self.act_steps]  # (n_envs, act_steps, action_dim)
                
                # Step EVAL environments (reset_within_step=True handles auto-reset for finished envs)
                obs, reward_venv, terminated_venv, truncated_venv, info_venv = self.eval_venv.step(action_chunk)
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
            
            # Collect results from all environments for this episode
            eval_episode_rewards.extend(episode_rewards.tolist())
            eval_episode_lengths.extend(episode_lengths.tolist())
        
        # Calculate metrics across all episodes (num_eval_episodes * eval_n_envs total)
        mean_episode_reward = np.mean(eval_episode_rewards)
        std_episode_reward = np.std(eval_episode_rewards)
        mean_episode_length = np.mean(eval_episode_lengths)
        
        # Calculate success rate  
        success_rate = np.mean(np.array(eval_episode_rewards) >= self.best_reward_threshold_for_success)
        
        # Log evaluation metrics
        log.info(f"Eval at step {cnt_train_step}: "
                f"reward={mean_episode_reward:.3f}±{std_episode_reward:.3f}, "
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
            }, step=cnt_train_step)

        self._log_eval_to_csv(
            step=cnt_train_step,
            eval_type='finetuned',
            mean_reward=mean_episode_reward,
            std_reward=std_episode_reward,
            success_rate=success_rate,
            mean_length=mean_episode_length,
            num_episodes=len(eval_episode_rewards)
        )
        # Switch back to training mode
        self.model.train()
        
        return mean_episode_reward, success_rate

    def run(self):

        # Start training loop
        timer = Timer()
        run_results = []
        cnt_train_step = 0
        done_venv = np.zeros((1, self.n_envs))
        
        while self.itr < self.n_train_itr:
            # Prepare video paths for training environments
            options_venv = [{} for _ in range(self.n_envs)]
            if self.itr % self.render_freq == 0 and self.render_video:
                for env_ind in range(self.n_render):
                    options_venv[env_ind]["video_path"] = os.path.join(
                        self.render_dir, f"itr-{self.itr}_trial-{env_ind}.mp4"
                    )

            # Always in training mode for the main loop
            self.model.train()

            # Reset env before iteration starts if specified or if done at the end of last iteration
            firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))
            if self.reset_at_iteration:
                prev_obs_venv = self.reset_env_all(options_venv=options_venv)
                firsts_trajs[0] = 1
            else:
                # if done at the end of last iteration, the envs are just reset
                firsts_trajs[0] = done_venv

            # Holder
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
            if self.save_full_observations:  # state-only
                obs_full_trajs = np.empty((0, self.n_envs, self.obs_dim))
                obs_full_trajs = np.vstack(
                    (obs_full_trajs, prev_obs_venv["state"][:, -1][None])
                )

            # Collect a set of trajectories from env
            for step in range(self.n_steps):
                # Run evaluation periodically using separate eval environments
                if self.run_eval and hasattr(self, 'eval_venv') and cnt_train_step % (self.eval_freq * self.n_envs * self.act_steps) == 0:
                    self.evaluate(cnt_train_step)

                if step % 10 == 0:
                    print(f"Processed step {step} of {self.n_steps}")

                # Select action
                with torch.no_grad():
                    cond = {
                        "state": torch.from_numpy(prev_obs_venv["state"])
                        .float()
                        .to(self.device)
                    }
                    samples, noises = self.model(
                        cond=cond,
                        deterministic=False,  # Always stochastic for training
                        return_chain=True,
                        return_noise=True,
                    )
                    output_venv = (
                        samples.trajectories.cpu().numpy()
                    )  # n_env x horizon x act
                    chains_venv = (
                        samples.chains.cpu().numpy()
                    )  # n_env x denoising x horizon x act
                    # noises_venv = (
                    #     noises.cpu().numpy()
                    # )
                action_venv = output_venv[:, : self.act_steps]

                # Apply multi-step action
                obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = (
                    self.venv.step(action_venv)
                )
                done_venv = terminated_venv | truncated_venv
                if self.save_full_observations:  # state-only
                    obs_full_venv = np.array(
                        [info["full_obs"]["state"] for info in info_venv]
                    )  # n_envs x act_steps x obs_dim
                    obs_full_trajs = np.vstack(
                        (obs_full_trajs, obs_full_venv.transpose(1, 0, 2))
                    )
                obs_trajs["state"][step] = prev_obs_venv["state"]
                chains_trajs[step] = chains_venv
                # noise_trajs[step] = noises_venv
                reward_trajs[step] = reward_venv
                terminated_trajs[step] = terminated_venv
                firsts_trajs[step + 1] = done_venv

                # update for next step
                prev_obs_venv = obs_venv

                # count steps --- not accounting for done within action chunk
                cnt_train_step += self.n_envs * self.act_steps

            # Summarize episode reward --- this needs to be handled differently depending on whether the environment is reset after each iteration. Only count episodes that finish within the iteration.
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
                if (
                    self.furniture_sparse_reward
                ):  # only for furniture tasks, where reward only occurs in one env step
                    episode_best_reward = episode_reward
                else:
                    episode_best_reward = np.array(
                        [
                            np.max(reward_traj) / self.act_steps
                            for reward_traj in reward_trajs_split
                        ]
                    )
                avg_episode_reward = np.mean(episode_reward)
                avg_best_reward = np.mean(episode_best_reward)
                success_rate = np.mean(
                    episode_best_reward >= self.best_reward_threshold_for_success
                )
            else:
                episode_reward = np.array([])
                num_episode_finished = 0
                avg_episode_reward = 0
                avg_best_reward = 0
                success_rate = 0
                log.info("[WARNING] No episode completed within the iteration!")

            # Update models
            with torch.no_grad():
                obs_trajs["state"] = (
                    torch.from_numpy(obs_trajs["state"]).float().to(self.device)
                )

                # Calculate value and logprobs - split into batches to prevent out of memory
                num_split = math.ceil(
                    self.n_envs * self.n_steps / self.logprob_batch_size
                )
                obs_ts = [{} for _ in range(num_split)]
                obs_k = einops.rearrange(
                    obs_trajs["state"],
                    "s e ... -> (s e) ...",
                )
                obs_ts_k = torch.split(obs_k, self.logprob_batch_size, dim=0)
                for i, obs_t in enumerate(obs_ts_k):
                    obs_ts[i]["state"] = obs_t
                values_trajs = np.empty((0, self.n_envs))
                for obs in obs_ts:
                    values = self.model.critic(obs).cpu().numpy().flatten()
                    values_trajs = np.vstack(
                        (values_trajs, values.reshape(-1, self.n_envs))
                    )
                chains_t = einops.rearrange(
                    torch.from_numpy(chains_trajs).float().to(self.device),
                    "s e t h d -> (s e) t h d",
                )
                chains_ts = torch.split(chains_t, self.logprob_batch_size, dim=0)
                logprobs_trajs = np.empty(
                    (
                        0,
                        self.model.ft_denoising_steps,
                        self.horizon_steps,
                        self.action_dim,
                    )
                )
                for obs, chains in zip(obs_ts, chains_ts):
                    logprobs = self.model.get_logprobs(obs, chains).cpu().numpy()
                    logprobs_trajs = np.vstack(
                        (
                            logprobs_trajs,
                            logprobs.reshape(-1, *logprobs_trajs.shape[1:]),
                        )
                    )

                # normalize reward with running variance if specified
                if self.reward_scale_running:
                    reward_trajs_transpose = self.running_reward_scaler(
                        reward=reward_trajs.T, first=firsts_trajs[:-1].T
                    )
                    reward_trajs = reward_trajs_transpose.T

                # bootstrap value with GAE if not terminal - apply reward scaling with constant if specified
                obs_venv_ts = {
                    "state": torch.from_numpy(obs_venv["state"])
                    .float()
                    .to(self.device)
                }
                advantages_trajs = np.zeros_like(reward_trajs)
                lastgaelam = 0
                for t in reversed(range(self.n_steps)):
                    if t == self.n_steps - 1:
                        nextvalues = (
                            self.model.critic(obs_venv_ts)
                            .reshape(1, -1)
                            .cpu()
                            .numpy()
                        )
                    else:
                        nextvalues = values_trajs[t + 1]
                    nonterminal = 1.0 - terminated_trajs[t]
                    # delta = r + gamma*V(st+1) - V(st)
                    delta = (
                        reward_trajs[t] * self.reward_scale_const
                        + self.gamma * nextvalues * nonterminal
                        - values_trajs[t]
                    )
                    # A = delta_t + gamma*lamdba*delta_{t+1} + ...
                    advantages_trajs[t] = lastgaelam = (
                        delta
                        + self.gamma * self.gae_lambda * nonterminal * lastgaelam
                    )
                returns_trajs = advantages_trajs + values_trajs

            # k for environment step
            obs_k = {
                "state": einops.rearrange(
                    obs_trajs["state"],
                    "s e ... -> (s e) ...",
                )
            }
            chains_k = einops.rearrange(
                torch.tensor(chains_trajs).float().to(self.device),
                "s e t h d -> (s e) t h d",
            )
            # noises_k = einops.rearrange(
            #     torch.tensor(noise_trajs).float().to(self.device),
            #     "s e t h d -> (s e) t h d",
            # )
            returns_k = (
                torch.tensor(returns_trajs).float().to(self.device).reshape(-1)
            )
            values_k = (
                torch.tensor(values_trajs).float().to(self.device).reshape(-1)
            )
            advantages_k = (
                torch.tensor(advantages_trajs).float().to(self.device).reshape(-1)
            )
            logprobs_k = torch.tensor(logprobs_trajs).float().to(self.device)

            # Update policy and critic
            total_steps = self.n_steps * self.n_envs
            inds_k = np.arange(total_steps)
            clipfracs = []
            for update_epoch in range(self.update_epochs):

                # for each epoch, go through all data in batches
                flag_break = False
                np.random.shuffle(inds_k)
                num_batch = max(1, total_steps // self.batch_size)  # skip last ones
                for batch in range(num_batch):
                    start = batch * self.batch_size
                    end = start + self.batch_size
                    inds_b = inds_k[start:end]  # b for batch
                    obs_b = {"state": obs_k["state"][inds_b]}
                    chains_b = chains_k[inds_b]
                    # noises_b = noises_k[inds_b]
                    returns_b = returns_k[inds_b]
                    values_b = values_k[inds_b]
                    advantages_b = advantages_k[inds_b]
                    logprobs_b = logprobs_k[inds_b]

                    # get loss
                    (
                        pg_loss,
                        entropy_loss,
                        v_loss,
                        clipfrac,
                        approx_kl,
                        ratio,
                        bc_loss,
                        eta,
                    ) = self.model.loss(
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

                    # update policy and critic
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
                    log.info(
                        f"approx_kl: {approx_kl}, update_epoch: {update_epoch}, num_batch: {num_batch}"
                    )

                    # Stop gradient update if KL difference reaches target
                    if self.target_kl is not None and approx_kl > self.target_kl:
                        flag_break = True
                        break
                if flag_break:
                    break

            # Explained variation of future rewards using value function
            y_pred, y_true = values_k.cpu().numpy(), returns_k.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = (
                np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            )

            # Plot state trajectories (only in D3IL)
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

            # Update lr, min_sampling_std
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

            # Log loss and save metrics
            run_results.append(
                {
                    "itr": self.itr,
                    "step": cnt_train_step,
                }
            )
            if self.save_trajs:
                run_results[-1]["obs_full_trajs"] = obs_full_trajs
                run_results[-1]["obs_trajs"] = obs_trajs
                run_results[-1]["chains_trajs"] = chains_trajs
                run_results[-1]["reward_trajs"] = reward_trajs
            if self.itr % self.log_freq == 0:
                time = timer()
                run_results[-1]["time"] = time
                # Always log training metrics (no more eval_mode in this loop)
                log.info(
                    f"{self.itr}: step {cnt_train_step:8d} | loss {loss:8.4f} | pg loss {pg_loss:8.4f} | value loss {v_loss:8.4f} | bc loss {bc_loss:8.4f} | reward {avg_episode_reward:8.4f} | eta {eta:8.4f} | t:{time:8.4f}"
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
                            "critic lr": self.critic_optimizer.param_groups[0][
                                "lr"
                            ],
                        },
                        step=cnt_train_step,
                        commit=True,
                    )
                run_results[-1]["train_episode_reward"] = avg_episode_reward
                with open(self.result_path, "wb") as f:
                    pickle.dump(run_results, f)
            self.itr += 1
