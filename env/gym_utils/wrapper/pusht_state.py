"""
Environment wrapper for PushT environment with state observations.

The PushT environment naturally terminates when success_threshold is reached,
so we don't need to handle success_steps_before_termination.
"""

import numpy as np
import gym
from gym import spaces
import os
import imageio
from env.pusht.pusht_env import PushTEnv


class PushTStateWrapper(gym.Env):
    def __init__(
        self,
        env=None,
        normalization_path=None,
        clamp_obs=False,
        init_state=None,
        render_hw=(96, 96),
        success_threshold=0.7695,
        **kwargs,
    ):
        # Create PushT environment if not provided
        if env is None:
            self.env = PushTEnv(
                render_size=render_hw[0],
                success_threshold=success_threshold
            )
        else:
            self.env = env
            
        self.init_state = init_state
        self.render_hw = render_hw
        self.clamp_obs = clamp_obs
        self.video_writer = None
        
        # Set up normalization
        print('success_threshold ', success_threshold)
        self.normalize = normalization_path is not None
        if self.normalize:
            normalization = np.load(normalization_path)
            self.obs_min = normalization["obs_min"]
            self.obs_max = normalization["obs_max"]
            self.action_min = normalization["action_min"]
            self.action_max = normalization["action_max"]
        
        # Setup action space - PushT actions are 2D (target position)
        # Normalized to [-1, 1]
        low = np.array([-1.0, -1.0], dtype=np.float32)
        high = np.array([1.0, 1.0], dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=low,
            high=high,
            shape=low.shape,
            dtype=low.dtype,
        )
        
        # Setup observation space
        # PushT state is 5D: [agent_x, agent_y, block_x, block_y, block_angle]
        self.observation_space = spaces.Dict()
        obs_dim = 5
        low = np.full(obs_dim, fill_value=-1.0, dtype=np.float32)
        high = np.full(obs_dim, fill_value=1.0, dtype=np.float32)
        self.observation_space["state"] = spaces.Box(
            low=low,
            high=high,
            shape=low.shape,
            dtype=np.float32,
        )
        
    def normalize_obs(self, obs):
        """Normalize observation to [-1, 1]"""
        obs = 2 * (
            (obs - self.obs_min) / (self.obs_max - self.obs_min + 1e-6) - 0.5
        )  # -> [-1, 1]
        if self.clamp_obs:
            obs = np.clip(obs, -1, 1)
        return obs
    
    def unnormalize_action(self, action):
        """Unnormalize action from [-1, 1] to original range"""
        action = (action + 1) / 2  # [-1, 1] -> [0, 1]
        return action * (self.action_max - self.action_min) + self.action_min
    
    def get_observation(self, raw_obs):
        """Convert raw observation to dict format"""
        obs = {"state": raw_obs.astype(np.float32)}
        if self.normalize:
            obs["state"] = self.normalize_obs(obs["state"])
        return obs
    
    def seed(self, seed=None):
        """Set random seed"""
        if seed is not None:
            np.random.seed(seed=seed)
            self.env.seed(seed)
        else:
            np.random.seed()
            
    def reset(self, options={}, **kwargs):
        """Reset environment"""
        # Close any existing video writer
        if self.video_writer is not None:
            self.video_writer.close()
            self.video_writer = None
        
        # Start video if specified
        if "video_path" in options:
            self.video_writer = imageio.get_writer(options["video_path"], fps=30)
        
        # Handle seeding:
        # - If seed explicitly provided in options, use it (for evaluation)
        # - Otherwise, generate a new random seed (for training)
        new_seed = options.get("seed", None)
        if new_seed is not None:
            # Explicit seed provided (evaluation mode)
            self.seed(seed=new_seed)
        else:
            # No seed provided - generate random seed for this reset (training mode)
            # This ensures each reset gets a different initial state during training
            random_seed = np.random.randint(0, 2**31 - 1)
            self.seed(seed=random_seed)
        
        # Reset to specific state if provided
        if self.init_state is not None:
            raw_obs = self.env.reset()
            self.env._set_state(self.init_state)
            raw_obs = self.env._get_obs()
        elif "init_state" in options:
            raw_obs = self.env.reset()
            self.env._set_state(options["init_state"])
            raw_obs = self.env._get_obs()
        else:
            # Random reset with the seed we just set
            raw_obs = self.env.reset()
            
        return self.get_observation(raw_obs)
    
    def step(self, action):
        """Step environment"""
        # Unnormalize action if needed
        if self.normalize:
            action = self.unnormalize_action(action)
            
        # Step the environment
        raw_obs, reward, done, info = self.env.step(action)
        obs = self.get_observation(raw_obs)
        
        # Record video frame if writer is active
        if self.video_writer is not None:
            frame = self.env.render(mode="rgb_array")
            self.video_writer.append_data(frame)
        
        # Close video writer when episode ends
        if done and self.video_writer is not None:
            self.video_writer.close()
            self.video_writer = None
        
        # PushT handles termination internally based on success_threshold
        # No need for additional termination logic
        
        return obs, reward, done, info
    
    def render(self, mode="rgb_array"):
        """Render environment"""
        if mode == "rgb_array":
            return self.env.render(mode=mode)
        else:
            return self.env.render(mode=mode)
    
    def close(self):
        """Close environment"""
        if self.video_writer is not None:
            self.video_writer.close()
            self.video_writer = None
        self.env.close()
    
    @property
    def max_episode_steps(self):
        """Maximum episode steps"""
        return 200  # Default max steps for PushT