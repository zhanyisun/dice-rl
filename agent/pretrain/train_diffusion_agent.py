"""
Pre-training diffusion policy

"""

import logging
import os
import wandb
import numpy as np
import torch

log = logging.getLogger(__name__)
from util.timer import Timer
from agent.pretrain.train_agent import PreTrainAgent, batch_to_device


class TrainDiffusionAgent(PreTrainAgent):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.max_episode_steps = cfg.env.max_episode_steps
        self.val_freq = 50


    def run(self):

        timer = Timer()
        self.epoch = 1
        for _ in range(self.n_epochs):

            options_venv = [{} for _ in range(self.n_envs)]
            if self.epoch % (2*self.val_freq) == 0:
                for env_ind in range(self.n_render):
                    options_venv[env_ind]["video_path"] = os.path.join(
                        self.render_dir, f"epoch-{self.epoch}_trial-{env_ind}.mp4"
                    )
                prev_obs_venv = self.reset_env_all(options_venv=options_venv)
                done_venv = np.zeros((1, self.n_envs))

            # train
            loss_train_epoch = []
            for batch_train in self.dataloader_train:
                if self.dataset_train.device == "cpu":
                    batch_train = batch_to_device(batch_train)

                self.model.train()
                loss_train = self.model.loss(*batch_train)
                loss_train.backward()
                loss_train_epoch.append(loss_train.item())

                self.optimizer.step()
                self.optimizer.zero_grad()
            loss_train = np.mean(loss_train_epoch)

            # validate
            loss_val_epoch = []
            if self.dataloader_val is not None and self.epoch % self.val_freq == 0:
                self.model.eval()
                for batch_val in self.dataloader_val:
                    if self.dataset_val.device == "cpu":
                        batch_val = batch_to_device(batch_val)
                    loss_val, infos_val = self.model.loss(*batch_val)
                    loss_val_epoch.append(loss_val.item())
                self.model.train()
            loss_val = np.mean(loss_val_epoch) if len(loss_val_epoch) > 0 else None

            # env rollout
            if self.epoch % (2*self.val_freq) == 0:
                log.info(f"Starting env rollout at epoch {self.epoch} (val_freq={self.val_freq})")
                
                policy = self.model
                policy.eval()
                
                dp_success_rate = self.evaluate_policy(policy, policy_name="dp")
                
                if self.use_wandb:
                    wandb.log(
                        {
                            "success rate - dp": dp_success_rate
                        },
                        step=self.epoch,
                        commit=False,
                    )
                
                self.model.train()
            # update lr
            self.lr_scheduler.step()

            # update ema
            self.step_ema()

            # save model
            if self.epoch % self.save_model_freq == 0 or self.epoch == self.n_epochs:
                self.save_model()

            # log loss
            if self.epoch % self.log_freq == 0:
                log.info(
                    f"{self.epoch}: train loss {loss_train:8.4f} | t:{timer():8.4f}"
                )
                if self.use_wandb:
                    if loss_val is not None:
                        wandb.log(
                            {"loss - val": loss_val}, step=self.epoch, commit=False
                        )
                    wandb.log(
                        {
                            "loss - train": loss_train,
                        },
                        step=self.epoch,
                        commit=True,
                    )

            # count
            self.epoch += 1
        wandb.finish()

    def evaluate_policy(self, policy, policy_name="policy"):
        """
        Generic evaluation method for any policy.
        
        Args:
            policy: The policy model to evaluate (must be in eval mode)
            policy_name: Name for logging and video files
            use_distilled: If True, use policy.get_distilled_action() instead of forward()
            
        Returns:
            success_rate: Success rate of the policy
        """
        # Reset environments for rollout
        options_venv = [{} for _ in range(self.n_envs)]
        for env_ind in range(self.n_render):
            options_venv[env_ind]["video_path"] = os.path.join(
                self.render_dir, f"epoch-{self.epoch}_{policy_name}_trial-{env_ind}.mp4"
            )
        prev_obs_venv = self.reset_env_all(options_venv=options_venv)
        done_venv = np.zeros(self.n_envs, dtype=bool)
        
        reward_trajs = []
        step_count = 0
        max_steps = self.max_episode_steps 

        while not np.all(done_venv) and step_count < max_steps // self.horizon_steps:
            step_count += 1
            # Select action
            with torch.no_grad():
                cond = {
                    "state": torch.from_numpy(prev_obs_venv["state"])
                    .float()
                    .to(self.device)
                }
                samples = policy(cond=cond, deterministic=True)
                output_venv = samples.trajectories.cpu().numpy()
                
            action_venv = output_venv[:, : self.act_steps]
            
            # Apply multi-step action
            obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = (
                self.venv.step(action_venv)
            )
            reward_trajs.append(reward_venv)
            done_venv = terminated_venv | truncated_venv
            prev_obs_venv = obs_venv
        
        if step_count >= max_steps // self.horizon_steps:
            log.warning(f"{policy_name} evaluation hit max steps limit at epoch {self.epoch}. Done envs: {np.sum(done_venv)}/{self.n_envs}")
        
        reward_trajs = np.stack(reward_trajs, axis=-1)
        reward_trajs = np.max(reward_trajs, axis=1)  # n_env x horizon
        success_rate = np.sum(
            reward_trajs >= self.best_reward_threshold_for_success
        ) / self.n_envs
        
        log.info(f"eval {policy_name}: success rate {success_rate:8.4f}")
        
        return success_rate
