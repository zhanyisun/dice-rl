"""
Environment wrapper for Robomimic environments with image observations.

Also return done=False since we do not terminate episode early.

Modified from https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/env/robomimic/robomimic_image_wrapper.py

"""

import numpy as np
import gym
from gym import spaces
import imageio
import xml.etree.ElementTree as ET
import os


def convert_10d_to_7d(action_10d):
    """
    Convert 10D action with 6D rotation to 7D action with axis-angle.
    
    Args:
        action_10d: (batch_size, 10) or (10,) with [pos(3), rot6d(6), gripper(1)]
    
    Returns:
        action_7d: Same shape but 7D with [pos(3), axis_angle(3), gripper(1)]
    """
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    from util.rotation_conversion import RotationTransformer
    
    rot_transformer = RotationTransformer(from_rep="rotation_6d", to_rep="axis_angle")
    
    # Handle both single action and batch of actions
    single_action = False
    if action_10d.ndim == 1:
        action_10d = action_10d[np.newaxis, :]
        single_action = True
    
    # Extract components
    pos = action_10d[:, :3]
    rot6d = action_10d[:, 3:9]
    gripper = action_10d[:, 9:10]
    
    # Convert 6D rotation to axis-angle
    axis_angle = rot_transformer.forward(rot6d)
    
    # Reconstruct 7D action
    action_7d = np.concatenate([pos, axis_angle, gripper], axis=-1)
    
    if single_action:
        action_7d = action_7d[0]
    
    return action_7d


class RobomimicImageWrapper(gym.Env):
    def __init__(
        self,
        env,
        shape_meta: dict,
        normalization_path=None,
        low_dim_keys=[
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        ],
        image_keys=[
            "agentview_image",
            "robot0_eye_in_hand_image",
        ],
        clamp_obs=False,
        init_state=None,
        render_hw=(256, 256),
        render_camera_name="robot0_eye_in_hand",
        success_steps_before_termination=5,
        use_6d_rot=False,  # Whether incoming actions use 6D rotation representation
    ):
        self.env = env
        self.init_state = init_state
        self.has_reset_before = False
        self.camera_modified = False
        self.render_hw = render_hw
        self.render_camera_name = render_camera_name
        self.video_writer = None
        self.clamp_obs = clamp_obs
        self.success_steps_before_termination = success_steps_before_termination
        self.use_6d_rot = use_6d_rot
        
        # Initialize tracking variables for success-based termination
        self.success_count = 0
        self.episode_reward = 0.0
        self.step_count = 0
        self.ever_succeeded = False

        # set up normalization
        self.normalize = normalization_path is not None
        if self.normalize:
            normalization = np.load(normalization_path)
            self.obs_min = normalization["obs_min"]
            self.obs_max = normalization["obs_max"]
            self.action_min = normalization["action_min"]
            self.action_max = normalization["action_max"]

        # setup spaces
        low = np.full(env.action_dimension, fill_value=-1)
        high = np.full(env.action_dimension, fill_value=1)
        self.action_space = gym.spaces.Box(
            low=low,
            high=high,
            shape=low.shape,
            dtype=low.dtype,
        )
        self.low_dim_keys = low_dim_keys
        self.image_keys = image_keys
        self.obs_keys = low_dim_keys + image_keys
        observation_space = spaces.Dict()
        for key, value in shape_meta["obs"].items():
            shape = value["shape"]
            if key.endswith("rgb"):
                min_value, max_value = 0, 1
            elif key.endswith("state"):
                min_value, max_value = -1, 1
            else:
                raise RuntimeError(f"Unsupported type {key}")
            this_space = spaces.Box(
                low=min_value,
                high=max_value,
                shape=shape,
                dtype=np.float32,
            )
            observation_space[key] = this_space
        self.observation_space = observation_space

    def normalize_obs(self, obs):
        obs = 2 * (
            (obs - self.obs_min) / (self.obs_max - self.obs_min + 1e-6) - 0.5
        )  # -> [-1, 1]
        if self.clamp_obs:
            obs = np.clip(obs, -1, 1)
        return obs

    def unnormalize_action(self, action):
        action = (action + 1) / 2  # [-1, 1] -> [0, 1]
        return action * (self.action_max - self.action_min) + self.action_min

    def get_observation(self, raw_obs):
        obs = {"rgb": None, "state": None}  # stack rgb if multiple cameras
        for key in self.obs_keys:
            if key in self.image_keys:
                raw_img = raw_obs[key]  # (H, W, 3) in BGR
                rgb_img = raw_img  # keep BGR for now since robomimic uses BGR
                if obs["rgb"] is None:
                    obs["rgb"] = rgb_img
                else:
                    obs["rgb"] = np.concatenate(
                        [obs["rgb"], rgb_img], axis=-1
                    )  # H W C
            else:
                if obs["state"] is None:
                    obs["state"] = raw_obs[key]
                else:
                    obs["state"] = np.concatenate([obs["state"], raw_obs[key]], axis=0)
        if self.normalize:
            obs["state"] = self.normalize_obs(obs["state"])
        obs["rgb"] = obs["rgb"].astype(np.uint8)
        return obs

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed=seed)
        else:
            np.random.seed()

    def reset(self, options={}, **kwargs):
        """Ignore passed-in arguments like seed"""
        # Close video if exists
        if self.video_writer is not None:
            self.video_writer.close()
            self.video_writer = None

        # Start video if specified
        if "video_path" in options:
            self.video_writer = imageio.get_writer(options["video_path"], fps=30)

        # Call reset
        new_seed = options.get(
            "seed", None
        )  # used to set all environments to specified seeds
        
        # Check if init_state is passed through options (for AsyncVectorEnv compatibility)
        init_state_from_options = options.get("init_state", None)
        
        effective_init_state = init_state_from_options if init_state_from_options is not None else self.init_state
        if effective_init_state is not None:
            if not self.has_reset_before:
                # the env must be fully reset at least once to ensure correct rendering
                self.env.reset()
                self.has_reset_before = True

            # always reset to the same state to be compatible with gym
            raw_obs = self.env.reset_to({"states": effective_init_state})
        elif new_seed is not None:
            self.seed(seed=new_seed)
            raw_obs = self.env.reset()
        else:
            # random reset
            raw_obs = self.env.reset()
        
        # Modified camera fov for tool hang
        # NOTE: If using default cam settings, bc+rl still works, but rl costs roughly 1.5x samples
        env_name = None
        if hasattr(self.env, 'env') and hasattr(self.env.env, '__class__'):
            env_name = self.env.env.__class__.__name__
        elif hasattr(self.env, '__class__'):
            env_name = self.env.__class__.__name__
        
        if env_name == 'ToolHang' and not self.camera_modified:
                print('Modifying camera for ToolHang environment (first reset only)...')
                current_state = self.env.env.sim.get_state().flatten()
                
                # Load modified XML  
                modified_xml_path = "cfg/robomimic/env_meta/tool_hang_model_modified.xml"
                if os.path.exists(modified_xml_path):
                    print(f"Loading pre-modified XML from {modified_xml_path}")
                    with open(modified_xml_path, "r") as f:
                        modified_xml = f.read()
                    
                    reset_state = {
                        "states": current_state,
                        "model": modified_xml
                    }

                    raw_obs = self.env.reset_to(reset_state)
                    self.camera_modified = True 
                else:
                    print(f"Warning: Modified XML not found at {modified_xml_path}, using original camera")
        
        # Reset tracking variables for new episode
        self.success_count = 0
        self.episode_reward = 0.0
        self.step_count = 0
        self.ever_succeeded = False
        
        return self.get_observation(raw_obs)

    def step(self, action):
        # If using 6D rotation, first unnormalize then convert to 7D
        if self.use_6d_rot:
            if action.shape[-1] != 10:
                raise ValueError(f"Expected 10D action when use_6d_rot=True, got {action.shape[-1]}D")
            
            # Unnormalize 10D action first (if needed)
            if self.normalize:   
                action = self.unnormalize_action(action)
                
            # Convert 10D (with 6D rotation) to 7D (with axis-angle)
            action = convert_10d_to_7d(action)
            
        else:
            # Standard path: unnormalize 7D action
            if self.normalize:
                action = self.unnormalize_action(action)
        
        raw_obs, reward, done, info = self.env.step(action)
        obs = self.get_observation(raw_obs)
        
        # Update tracking variables
        self.step_count += 1
        self.episode_reward += reward
        terminated = False
        
        # Check for success-based termination
        if reward > 0:  # Success detected
            if not hasattr(self, 'success_count'):
                self.success_count = 0
            self.success_count += 1
            self.ever_succeeded = True  # Track if we ever succeeded
            if self.success_count >= self.success_steps_before_termination:  # Terminate after configured steps
                terminated = True
                # print(f"DEBUG: Episode terminating at step {self.step_count} with total_reward={self.episode_reward:.1f}")
                self.success_count = 0
                self.episode_reward = 0.0
                self.step_count = 0
                self.ever_succeeded = False
            else:
                terminated = False
        else:
            # Only reset success count if we haven't succeeded yet (enforce "once success, always success")
            if not hasattr(self, 'ever_succeeded'):
                self.ever_succeeded = False
            
            if not self.ever_succeeded:
                self.success_count = 0
            # else:
            #     print(f"WARNING: Got reward={reward} after success at step {self.step_count}! Not resetting success_count={self.success_count}")
            terminated = False
            
        # render if specified
        if self.video_writer is not None:
            video_img = self.render(mode="rgb_array")
            self.video_writer.append_data(video_img)

        return obs, reward, terminated, info

    def render(self, mode="rgb_array"):
        h, w = self.render_hw
        return self.env.render(
            mode=mode,
            height=h,
            width=w,
            camera_name=self.render_camera_name,
        )


if __name__ == "__main__":
    import os
    from omegaconf import OmegaConf
    import json

    os.environ["MUJOCO_GL"] = "egl"

    cfg = OmegaConf.load("cfg/robomimic/pretrain/tool_hang/pre_flow_matching_mlp_img.yaml")
    shape_meta = cfg["shape_meta"]

    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.obs_utils as ObsUtils
    import matplotlib.pyplot as plt

    wrappers = cfg.env.wrappers
    obs_modality_dict = {
        "low_dim": (
            wrappers.robomimic_image.low_dim_keys
            if "robomimic_image" in wrappers
            else wrappers.robomimic_lowdim.low_dim_keys
        ),
        "rgb": (
            wrappers.robomimic_image.image_keys
            if "robomimic_image" in wrappers
            else None
        ),
    }
    if obs_modality_dict["rgb"] is None:
        obs_modality_dict.pop("rgb")
    ObsUtils.initialize_obs_modality_mapping_from_dict(obs_modality_dict)

    with open(cfg.robomimic_env_cfg_path, "r") as f:
        env_meta = json.load(f)
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=False,
        use_image_obs=True,
    )
    env.env.hard_reset = False

    wrapper = RobomimicImageWrapper(
        env=env,
        shape_meta=shape_meta,
        image_keys=["robot0_eye_in_hand_image"],
    )
    wrapper.seed(0)
    obs = wrapper.reset()
    print(obs.keys(), obs['rgb'].shape)
    img = wrapper.render()
    wrapper.close()
    plt.imshow(img)
    plt.savefig("test_0_change_view.png")
