"""Pre-training flow matching policy."""

import logging
import os
import wandb
import numpy as np
import torch

log = logging.getLogger(__name__)
from util.timer import Timer
from agent.pretrain.train_agent import PreTrainAgent, batch_to_device


class TrainFlowMatchingAgent(PreTrainAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.max_episode_steps = cfg.env.max_episode_steps
        self.val_freq = 100


    def run(self):

        timer = Timer()
        self.epoch = 1
        for _ in range(self.n_epochs):


            # train
            loss_train_epoch = []
            flow_loss_epoch = []
            distill_loss_epoch = []
            
            for batch_train in self.dataloader_train:
                if self.dataset_train.device == "cpu":
                    batch_train = batch_to_device(batch_train)

                self.model.train()
                loss_result = self.model.loss(*batch_train)
                
                # Handle both dictionary and scalar loss returns
                if isinstance(loss_result, dict):
                    loss_train = loss_result["total_loss"]
                    # Track individual losses for logging
                    flow_loss = loss_result.get("flow_loss", loss_train)
                    distill_loss = loss_result.get("distill_loss", 0.0)
                    flow_loss_epoch.append(flow_loss.item() if hasattr(flow_loss, 'item') else flow_loss)
                    distill_loss_epoch.append(distill_loss.item() if hasattr(distill_loss, 'item') else distill_loss)
                else:
                    loss_train = loss_result
                    flow_loss = loss_train
                    distill_loss = 0.0
                    flow_loss_epoch.append(flow_loss.item())
                    distill_loss_epoch.append(0.0)
                
                loss_train.backward()
                loss_train_epoch.append(loss_train.item())

                self.optimizer.step()
                self.optimizer.zero_grad()
            loss_train = np.mean(loss_train_epoch)
            mean_flow_loss = np.mean(flow_loss_epoch)
            mean_distill_loss = np.mean(distill_loss_epoch)

            # validate
            loss_val_epoch = []
            if self.dataloader_val is not None and self.epoch % self.val_freq == 0:
                self.model.eval()
                for batch_val in self.dataloader_val:
                    if self.dataset_val.device == "cpu":
                        batch_val = batch_to_device(batch_val)
                    loss_result = self.model.loss(*batch_val)
                    
                    # Handle both dictionary and scalar loss returns
                    if isinstance(loss_result, dict):
                        loss_val = loss_result["total_loss"]
                    else:
                        loss_val = loss_result
                    
                    loss_val_epoch.append(loss_val.item())
                self.model.train()
            loss_val = np.mean(loss_val_epoch) if len(loss_val_epoch) > 0 else None

            # env rollout
            if self.epoch % self.val_freq == 0:
                log.info(f"Starting env rollout at epoch {self.epoch} (val_freq={self.val_freq})")
                
                policy = self.model
                policy.eval()
                
                flow_success_rate = self.evaluate_policy(policy, policy_name="flow")
                
                if self.use_wandb:
                    wandb.log(
                        {
                            "success rate - flow": flow_success_rate
                        },
                        step=self.epoch,
                        commit=False,
                    )
                
                self.model.train()
            # update lr
            self.lr_scheduler.step()

            self.step_ema()

            # save model
            if self.epoch % self.save_model_freq == 0 or self.epoch == self.n_epochs:
                self.save_model()

            # log loss
            if self.epoch % self.log_freq == 0:
                # Log with more detail if using distillation
                if mean_distill_loss > 0:
                    log.info(
                        f"{self.epoch}: total loss {loss_train:8.4f} | flow {mean_flow_loss:8.4f} | distill {mean_distill_loss:8.4f} | t:{timer():8.4f}"
                    )
                else:
                    log.info(
                        f"{self.epoch}: train loss {loss_train:8.4f} | t:{timer():8.4f}"
                    )
                    
                if self.use_wandb:
                    log_dict = {"loss - train": loss_train}
                    
                    if loss_val is not None:
                        log_dict["loss - val"] = loss_val
                    
                    # Add individual losses if using distillation
                    if mean_distill_loss > 0:
                        log_dict["loss - flow"] = mean_flow_loss
                        log_dict["loss - distill"] = mean_distill_loss
                    
                    wandb.log(log_dict, step=self.epoch, commit=True)

            # count
            self.epoch += 1
        wandb.finish()

    def evaluate_policy(self, policy, policy_name="policy"):
        """
        Generic evaluation method for policy.
        
        Args:
            policy: The policy model to evaluate (must be in eval mode)
            policy_name: Name for logging and video files
            
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
                
                # Use flow matching policy
                samples = policy(cond=cond)
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
