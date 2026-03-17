"""
Environment wrapper for PushT environment with image observations.

The PushT environment naturally terminates when success_threshold is reached,
so we don't need to handle success_steps_before_termination.
"""

import numpy as np
import gym
from gym import spaces
import os
import imageio
from env.pusht.pusht_image_env import PushTImageEnv


class PushTImageWrapper(gym.Env):
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
        print('success_threshold ', success_threshold)
        # Create PushT environment if not provided
        if env is None:
            self.env = PushTImageEnv(
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
        self.normalize = normalization_path is not None
        if self.normalize:
            normalization = np.load(normalization_path)
            self.obs_min = normalization["obs_min"][:2]
            self.obs_max = normalization["obs_max"][:2]
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
        
        # Setup observation space with both state and image
        self.observation_space = spaces.Dict()
        
        # State observation: 5D [agent_x, agent_y, block_x, block_y, block_angle]
        obs_dim = 2
        state_low = np.full(obs_dim, fill_value=-1.0, dtype=np.float32)
        state_high = np.full(obs_dim, fill_value=1.0, dtype=np.float32)
        self.observation_space["state"] = spaces.Box(
            low=state_low,
            high=state_high,
            shape=state_low.shape,
            dtype=np.float32,
        )
        
        # Image observation: (H, W, 3) RGB image in range [0, 255]
        # Note: We keep it as (H, W, C) format and uint8 with [0, 255] range
        self.observation_space["rgb"] = spaces.Box(
            low=0,
            high=255,
            shape=(self.render_hw[0], self.render_hw[1], 3),
            dtype=np.uint8,
        )
        
    def normalize_obs(self, obs):
        """Normalize observation to [-1, 1]"""
        # print('self.obs_min ', self.obs_min)
        # print('self.obs_max ', self.obs_max)
        # print('obs before norm', obs)
        obs = 2 * (
            (obs - self.obs_min) / (self.obs_max - self.obs_min + 1e-6) - 0.5
        )  # -> [-1, 1]
        # print('obs after norm', obs)
        # if obs.min() < -1.3:
        #     print(f"WARNING: PushTImageWrapper normalized obs has min {obs.min()}")
        #     print(f"  raw obs was: {obs}")
        #     print(f"  obs_min: {self.obs_min}, obs_max: {self.obs_max}")
        if self.clamp_obs:
            obs = np.clip(obs, -1, 1)
        return obs
    
    def unnormalize_action(self, action):
        """Unnormalize action from [-1, 1] to original range"""
        action = (action + 1) / 2  # [-1, 1] -> [0, 1]
        return action * (self.action_max - self.action_min) + self.action_min
    
    def get_observation(self, raw_obs):
        """Convert raw observation to dict format with state and image"""
        # PushTImageEnv now returns {'rgb': ..., 'state': ...}
        # 'state' is 2D (agent_pos)
        state = raw_obs['state'].astype(np.float32)
        # print('raw state ', state)
        if self.normalize:
            # Only normalize the 2D agent position
            state = self.normalize_obs(state)
            
        # Get image from raw_obs - already rendered by PushTImageEnv
        # PushT renders as (H, W, 3) uint8 in range [0, 255]
        rgb = raw_obs['rgb']
        
        # The env already returns (H, W, 3) format
        # Just ensure it's uint8 and in corr
        # ect range
        assert(rgb.dtype == np.uint8)
        assert(rgb.max() > 1.0)
        # Keep as (H, W, C) format - this is what the training code expects
        obs = {
            "state": state,
            "rgb": rgb  # (H, W, 3) format
        }
        
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
            # Use the image from raw_obs instead of calling render again
            frame = raw_obs['rgb']
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