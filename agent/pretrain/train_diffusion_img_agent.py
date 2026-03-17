"""Pre-training diffusion policy with image observations."""

import logging
import os
import wandb
import numpy as np
import torch
import einops

log = logging.getLogger(__name__)
from util.timer import Timer
from agent.pretrain.train_diffusion_agent import TrainDiffusionAgent
from agent.pretrain.train_agent import batch_to_device


class TrainDiffusionImgAgent(TrainDiffusionAgent):
    """
    Diffusion pretraining agent for image observations.
    
    This extends TrainDiffusionAgent to properly handle both state and RGB observations
    during environment rollouts.
    """
    
    def __init__(self, cfg):
        # Validate that use_6d_rot and abs_action are used together
        use_6d_rot = cfg.get('use_6d_rot', False)
        abs_action = cfg.get('abs_action', False)
        
        # Ensure they are either both true or both false
        if use_6d_rot != abs_action:
            raise ValueError(
                f"use_6d_rot and abs_action must be used together. "
                f"Got use_6d_rot={use_6d_rot}, abs_action={abs_action}. "
                f"Set both to True for 6D rotation with absolute actions, or both to False."
            )
        
        if use_6d_rot and abs_action:
            assert cfg.get('action_dim', 0) == 10, (
                f"When use_6d_rot=True, action_dim must be 10 (pos:3 + rot6d:6 + gripper:1). "
                f"Got action_dim={cfg.get('action_dim', 0)}"
            )
            
            paths_to_check = [
                ('train_dataset_path', cfg.get('train_dataset_path', '')),
                ('normalization_path', cfg.get('normalization_path', '')),
                ('robomimic_env_cfg_path', cfg.get('robomimic_env_cfg_path', ''))
            ]
            
            for path_name, path_value in paths_to_check:
                assert 'abs' in path_value.lower(), (
                    f"When abs_action=True, {path_name} should contain 'abs'. Got: {path_value}"
                )
        
        super().__init__(cfg)
        
        shape_meta = cfg.shape_meta
        self.obs_dims = {k: shape_meta.obs[k]["shape"] for k in shape_meta.obs}
        log.info(f"TrainDiffusionImgAgent initialized with obs_dims: {self.obs_dims.keys()}")
        self.val_freq = 50

    def run(self):
        timer = Timer()
        self.epoch = 1
        for _ in range(self.n_epochs):
            # train
            loss_train_epoch = []
            for batch_idx, batch_train in enumerate(self.dataloader_train):
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
                    loss_val = self.model.loss(*batch_val)
                    loss_val_epoch.append(loss_val.item())
                self.model.train()
            loss_val = np.mean(loss_val_epoch) if len(loss_val_epoch) > 0 else None

            # env rollout with image observations
            if self.epoch % (2*self.val_freq) == 0:
                log.info(f"Starting env rollout at epoch {self.epoch} (val_freq={self.val_freq})")
                # Reset environments for rollout
                options_venv = [{} for _ in range(self.n_envs)]
                for env_ind in range(self.n_render):
                    options_venv[env_ind]["video_path"] = os.path.join(
                        self.render_dir, f"epoch-{self.epoch}_trial-{env_ind}.mp4"
                    )
                prev_obs_venv = self.reset_env_all(options_venv=options_venv)
                done_venv = np.zeros(self.n_envs, dtype=bool)
                
                policy = self.model
                policy.eval()
                reward_trajs = []
                step_count = 0
                max_steps = self.max_episode_steps
                while not np.all(done_venv) and step_count < max_steps // self.horizon_steps:
                    step_count += 1
                    with torch.no_grad():
                        cond = {}
                        for key in self.obs_dims:
                            obs_data = torch.from_numpy(prev_obs_venv[key]).float().to(self.device)
                            # Handle RGB format conversion from (H, W, C) to (C, H, W)
                            # Environment provides (N, T, H, W, C) but model expects (N, T, C, H, W)
                            if key == "rgb" and len(obs_data.shape) == 5:
                                obs_data = obs_data.permute(0, 1, 4, 2, 3)  # (N, T, H, W, C) -> (N, T, C, H, W)                            
                            cond[key] = obs_data

                        samples = policy(
                            cond=cond,
                            deterministic=True
                        )
                        output_venv = (
                            samples.trajectories.cpu().numpy()
                        )  # n_env x horizon x act
                    action_venv = output_venv[:, : self.act_steps]

                    # Apply multi-step action
                    obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = (
                        self.venv.step(action_venv)
                    )

                    reward_trajs.append(reward_venv)
                    done_venv = terminated_venv | truncated_venv
                    prev_obs_venv = obs_venv
                
                if step_count >= max_steps // self.horizon_steps:
                    log.warning(f"Evaluation rollout hit max steps limit at epoch {self.epoch}. Done envs: {np.sum(done_venv)}/{self.n_envs}")
                
                reward_trajs = np.stack(reward_trajs, axis=-1)
                reward_trajs = np.max(reward_trajs, axis=1)  # n_env x horizon
                success_rate = np.sum(
                    reward_trajs >= self.best_reward_threshold_for_success
                ) / self.n_envs
                log.info(
                    f"eval: success rate {success_rate:8.4f} "
                )
                if self.use_wandb:
                    wandb.log(
                        {
                            "success rate - eval": success_rate
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