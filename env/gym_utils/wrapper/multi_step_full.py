"""
MultiStepFull wrapper for collecting all intermediate observations and rewards.

Key features:
- Inherits from MultiStep for standard functionality (observation/action spaces)
- Overrides step() to add full trajectory collection in info dict
- Handles initial observation o_0 from reset
- Handles early termination with partial trajectories  
- Provides all intermediate data for upsampling
"""

import gym
from typing import Optional
from gym import spaces
import numpy as np
from collections import defaultdict, deque

from env.gym_utils.wrapper.multi_step import MultiStep, aggregate

class MultiStepFull(MultiStep):
    """
    Environment wrapper that collects all intermediate observations and rewards.
    
    Inherits from MultiStep to get observation_space and action_space properties.
    Overrides __init__, reset(), and step() to add full trajectory tracking.
    
    Trajectory structure: o_0, a_0, r_0, o_1, a_1, r_1, ..., o_T
    Where o_0 is from reset(), and (a_t, r_t, o_{t+1}) are from step(a_t)
    """
    
    def __init__(self, env, n_obs_steps=1, n_action_steps=8, max_episode_steps=700, reset_within_step=True, **kwargs):
        super().__init__(
            env=env,
            n_obs_steps=n_obs_steps, 
            n_action_steps=n_action_steps,
            max_episode_steps=max_episode_steps,
            reset_within_step=reset_within_step,
            **kwargs
        )
        
        # Additional tracking for full trajectory
        self.initial_obs = None
        self.is_first_step = True  # Flag for first step after reset
        
    def reset(self, **kwargs):
        """Reset environment and store initial observation."""
        obs = super().reset(**kwargs)  # Parent only returns obs, not info
        
        self.is_first_step = True
        
        # Store o_0 for first chunk - extract from stacked obs
        # obs has shape (n_obs_steps, ...) and we want the last one (most recent)
        if isinstance(obs, dict):
            # Dict observation - extract last timestep for each key
            self.initial_obs = {k: v[-1] for k, v in obs.items()}
        else:
            # Box observation - extract last timestep
            self.initial_obs = obs[-1]
        
        return obs  # Return same as parent
    
    def step(self, action):
        """
        Execute action chunk and collect all intermediate data.
        
        Overrides parent step() to add full trajectory collection.
        """
        # Handle action shape 
        if action.ndim == 1:  # in case action_steps = 1
            action = action[None]
            
        # Storage for trajectory data
        step_observations = []  # [o_{t+1}, ..., o_{t+k}]
        step_rewards = []       # [r_t, ..., r_{t+k-1}]  
        step_actions = []       # [a_t, ..., a_{t+k-1}]
        step_dones = []         # [done_t, ..., done_{t+k-1}]
        
        # Determine if we include o_0
        include_initial = self.is_first_step
        initial_obs_to_include = self.initial_obs if include_initial else None

        truncated = False       
        terminated = False
        # Execute action chunk (mirroring parent's logic)
        for act_step, act in enumerate(action):
            self.cnt += 1
            if terminated or truncated:
                break

            # Step environment (gym returns 4 values, not 5 like gymnasium)
            observation, reward, done, info = self.env.step(act)
            
            self.obs.append(observation)
            
            # Store for full trajectory
            step_observations.append(observation)
            step_rewards.append(reward)
            step_actions.append(act)
            
            if "TimeLimit.truncated" not in info:
                if done:
                    terminated = True
                elif (
                    self.max_episode_steps is not None
                ) and self.cnt >= self.max_episode_steps:
                    truncated = True
            else:
                truncated = info["TimeLimit.truncated"]
                terminated = done
            done = truncated or terminated
            # Store done flag for full trajectory
            step_dones.append(done)
        
        # No longer first step
        self.is_first_step = False
        # Get observation using parent's method
        observation = self._get_obs(self.n_obs_steps)
        
        # Aggregate rewards and done from our collected data
        reward = aggregate(step_rewards, self.reward_agg_method, self.gamma)
        done = aggregate(step_dones, "max")
        
        # Create info with full trajectory data
        info = {}
        info['full_trajectory'] = {
            'initial_obs': initial_obs_to_include,  # o_0 if first chunk, else None
            'observations': step_observations,      # Length k: [o_1, ..., o_k]
            'rewards': step_rewards,                # Length k: [r_0, ..., r_{k-1}]
            'actions': step_actions,                # Length k: [a_0, ..., a_{k-1}]
            'dones': step_dones,                    # Length k: [done_0, ..., done_{k-1}]
            'include_initial': include_initial,
        }
        
        # Handle reset within step
        if self.reset_within_step and step_dones[-1]:
            # Reset automatically sets self.initial_obs and self.is_first_step
            observation = self.reset()
                    
        return observation, reward, terminated, truncated, info
    