"""
Parent pre-training agent class.

"""

import os
import random
import numpy as np
from omegaconf import OmegaConf
import torch
import hydra
import logging
import wandb
from copy import deepcopy
from torch.nn.modules.batchnorm import _BatchNorm

log = logging.getLogger(__name__)
from util.scheduler import CosineAnnealingWarmupRestarts
from env.gym_utils import make_async

DEVICE = "cuda:0"


def to_device(x, device=DEVICE):
    if torch.is_tensor(x):
        return x.to(device)
    elif type(x) is dict:
        return {k: to_device(v, device) for k, v in x.items()}
    else:
        print(f"Unrecognized type in `to_device`: {type(x)}")


def batch_to_device(batch, device="cuda:0"):
    vals = [to_device(getattr(batch, field), device) for field in batch._fields]
    return type(batch)(*vals)

class EMA:
    """
    Exponential Moving Average with warmup and adaptive decay.
    Interface-compatible with previous EMA implementation.
    """

    def __init__(self, cfg):
        """
        Args:
            cfg.decay: max_value (float), the maximum EMA decay.
            cfg.update_after_step (int): number of steps to skip before EMA starts.
            cfg.inv_gamma (float): EMA warmup inverse gamma (default 1.0)
            cfg.power (float): EMA warmup power (default 2/3)
            cfg.min_value (float): minimum EMA decay (default 0.0)
        """
        self.update_after_step = getattr(cfg, "update_after_step", 0)
        self.inv_gamma = getattr(cfg, "inv_gamma", 1.0)
        self.power = getattr(cfg, "power", 0.75)
        self.min_value = getattr(cfg, "min_value", 0.0)
        self.max_value = getattr(cfg, "decay", 0.9999)

        self.optimization_step = 0
        self.decay = 0.0

    def get_decay(self):
        step = max(0, self.optimization_step - self.update_after_step - 1)
        value = 1 - (1 + step / self.inv_gamma) ** -self.power

        if step <= 0:
            return 0.0
        return max(self.min_value, min(value, self.max_value))

    @torch.no_grad()
    def update_model_average(self, ma_model, current_model):
        """
        Perform one EMA update step.
        """
        self.decay = self.get_decay()

        for module, ema_module in zip(current_model.modules(), ma_model.modules()):
            for param, ema_param in zip(
                module.parameters(recurse=False), ema_module.parameters(recurse=False)
            ):
                if isinstance(param, dict):
                    raise RuntimeError("Dict parameter not supported")

                if isinstance(module, _BatchNorm):
                    # skip EMA for BatchNorm
                    ema_param.copy_(param.to(dtype=ema_param.dtype).data)
                elif not param.requires_grad:
                    # skip frozen weights
                    ema_param.copy_(param.to(dtype=ema_param.dtype).data)
                else:
                    ema_param.mul_(self.decay)
                    ema_param.add_(
                        param.data.to(dtype=ema_param.dtype), alpha=1 - self.decay
                    )

        self.optimization_step += 1

class PreTrainAgent:

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = cfg.device
        self.seed = cfg.get("seed", 42)
        
        if hasattr(cfg, 'abs_action') and cfg.abs_action:
            # Check train_dataset_path if it exists
            if hasattr(cfg, 'train_dataset_path'):
                train_dataset_path = cfg.get('train_dataset_path', '')
                assert 'abs' in train_dataset_path.lower(), (
                    f"When abs_action=True, train_dataset_path should contain 'abs' to indicate absolute action dataset. "
                    f"Got: {train_dataset_path}"
                )
            
            # Check normalization_path if it exists
            if hasattr(cfg, 'normalization_path'):
                normalization_path = cfg.get('normalization_path', '')
                assert 'abs' in normalization_path.lower(), (
                    f"When abs_action=True, normalization_path should contain 'abs' to indicate absolute action normalization. "
                    f"Got: {normalization_path}"
                )
            
            # Check robomimic_env_cfg_path if it exists
            if hasattr(cfg, 'robomimic_env_cfg_path'):
                robomimic_env_cfg_path = cfg.get('robomimic_env_cfg_path', '')
                assert 'abs' in robomimic_env_cfg_path.lower(), (
                    f"When abs_action=True, robomimic_env_cfg_path should contain 'abs' to indicate absolute action environment config. "
                    f"Got: {robomimic_env_cfg_path}"
                )

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # CUDA deterministic settings
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)

        # Wandb
        self.use_wandb = cfg.wandb is not None
        if cfg.wandb is not None:
            wandb.init(
                project=cfg.wandb.project,
                name=cfg.wandb.run,
                config=OmegaConf.to_container(cfg, resolve=True),
            )

        # Make vectorized env
        self.env_name = cfg.env.name
        env_type = cfg.env.get("env_type", None)
        self.venv = make_async(
            cfg.env.name,
            env_type=env_type,
            num_envs=cfg.env.n_envs,
            asynchronous=True,
            max_episode_steps=cfg.env.max_episode_steps,
            wrappers=cfg.env.get("wrappers", None),
            robomimic_env_cfg_path=cfg.get("robomimic_env_cfg_path", None),
            shape_meta=cfg.get("shape_meta", None),
            use_image_obs=cfg.env.get("use_image_obs", False),
            render=cfg.env.get("render", False),
            render_offscreen=cfg.env.get("save_video", False),
            normalization_path=cfg.get("normalization_path", None),
            obs_dim=cfg.obs_dim,
            action_dim=cfg.action_dim,
            abs_action=cfg.get("abs_action", False),  # Pass absolute action flag
            **cfg.env.specific if "specific" in cfg.env else {},
        )
        if not env_type == "furniture":
            self.venv.seed(
                [self.seed + i for i in range(cfg.env.n_envs)]
            )  # otherwise parallel envs might have the same initial states!
            # isaacgym environments do not need seeding
        self.n_envs = cfg.env.n_envs
        self.n_cond_step = cfg.cond_steps
        self.obs_dim = cfg.obs_dim
        self.action_dim = cfg.action_dim
        self.act_steps = cfg.act_steps
        self.horizon_steps = cfg.horizon_steps

        # Build model
        self.model = hydra.utils.instantiate(cfg.model)
        self.ema = EMA(cfg.ema)
        self.ema_model = deepcopy(self.model)

        # Training params
        self.n_epochs = cfg.train.n_epochs
        self.batch_size = cfg.train.batch_size
        self.update_ema_freq = cfg.train.update_ema_freq
        self.epoch_start_ema = cfg.train.epoch_start_ema
        self.val_freq = cfg.train.get("val_freq", 100)
        self.best_reward_threshold_for_success = (
            len(self.venv.pairs_to_assemble)
            if env_type == "furniture"
            else cfg.env.best_reward_threshold_for_success
        )

        # Logging, checkpoints
        self.logdir = cfg.logdir
        self.render_dir = os.path.join(self.logdir, "render")

        self.checkpoint_dir = os.path.join(self.logdir, "checkpoint")
        os.makedirs(self.render_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.log_freq = cfg.train.get("log_freq", 1)
        self.save_model_freq = cfg.train.save_model_freq
        
        self.render_freq = cfg.train.render.freq
        self.n_render = cfg.train.render.num
        self.render_video = cfg.env.get("save_video", False)
        assert self.n_render <= self.n_envs, "n_render must be <= n_envs"
        assert not (
            self.n_render <= 0 and self.render_video
        ), "Need to set n_render > 0 if saving video"

        # Build dataset
        self.dataset_train = hydra.utils.instantiate(cfg.train_dataset)
        self.dataloader_train = torch.utils.data.DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=4 if self.dataset_train.device == "cpu" else 0,
            shuffle=True,
            pin_memory=True if self.dataset_train.device == "cpu" else False,
            persistent_workers=False
        )
        self.dataloader_val = None
        if "train_split" in cfg.train and cfg.train.train_split < 1:
            val_indices = self.dataset_train.set_train_val_split(cfg.train.train_split)
            self.dataset_val = deepcopy(self.dataset_train)
            self.dataset_val.set_indices(val_indices)
            self.dataloader_val = torch.utils.data.DataLoader(
                self.dataset_val,
                batch_size=self.batch_size,
                num_workers=4 if self.dataset_val.device == "cpu" else 0,
                shuffle=True,
                pin_memory=True if self.dataset_val.device == "cpu" else False,
                persistent_workers=False
            )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.train.learning_rate,
            weight_decay=cfg.train.weight_decay,
        )
        self.lr_scheduler = CosineAnnealingWarmupRestarts(
            self.optimizer,
            first_cycle_steps=cfg.train.lr_scheduler.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=cfg.train.learning_rate,
            min_lr=cfg.train.lr_scheduler.min_lr,
            warmup_steps=cfg.train.lr_scheduler.warmup_steps,
            gamma=1.0,
        )
        self.reset_parameters()

    def run(self):
        raise NotImplementedError

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        self.ema.update_model_average(self.ema_model, self.model)

    def save_model(self):
        """
        saves model and ema to disk;
        """
        data = {
            "epoch": self.epoch,
            "model": self.model.state_dict(),
            "ema": self.ema_model.state_dict(),
        }
        savepath = os.path.join(self.checkpoint_dir, f"state_{self.epoch}.pt")
        torch.save(data, savepath)
        log.info(f"Saved model to {savepath}")

    def load(self, epoch):
        """
        loads model and ema from disk
        """
        loadpath = os.path.join(self.checkpoint_dir, f"state_{epoch}.pt")
        data = torch.load(loadpath, weights_only=True)

        self.epoch = data["epoch"]
        self.model.load_state_dict(data["model"])
        self.ema_model.load_state_dict(data["ema"])

    def reset_env_all(self, verbose=False, options_venv=None, **kwargs):
        if options_venv is None:
            options_venv = [
                {k: v for k, v in kwargs.items()} for _ in range(self.n_envs)
            ]
        obs_venv = self.venv.reset_arg(options_list=options_venv)
        if isinstance(obs_venv, list):
            obs_venv = {
                key: np.stack([obs_venv[i][key] for i in range(self.n_envs)])
                for key in obs_venv[0].keys()
            }
        if verbose:
            for index in range(self.n_envs):
                logging.info(
                    f"<-- Reset environment {index} with options {options_venv[index]}"
                )
        return obs_venv

    def reset_env(self, env_ind, verbose=False):
        task = {}
        obs = self.venv.reset_one_arg(env_ind=env_ind, options=task)
        if verbose:
            logging.info(f"<-- Reset environment {env_ind} with task {task}")
        return obs
