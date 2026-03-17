"""
State trajectory visualization utilities for analyzing robot policy behavior.

This module visualizes robot end-effector trajectories directly in 3D space
without dimensionality reduction, providing clear and interpretable visualizations.
"""

import os
import numpy as np
import torch
import plotly.graph_objects as go
import plotly.express as px
import logging
import wandb
from typing import List, Dict, Optional, Tuple

log = logging.getLogger(__name__)


class StateTrajectoryVisualizer:
    """Visualize robot end-effector trajectories in 3D space."""
    
    def __init__(self, **kwargs):
        """
        Initialize the visualizer.
        Direct 3D plotting of robot end-effector positions.
        """
        # Robot state indices for our saved trajectory data
        # We only save [0:3] and [9:12] from original state, so in our saved data:
        self.robot0_eef_pos_idx = slice(0, 3)    # robot0_eef_pos [0:3] - xyz
        self.robot1_eef_pos_idx = slice(3, 6)    # robot1_eef_pos [3:6] - xyz (was [9:12] in full state)
    
    def extract_eef_positions(self, trajectories: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Extract robot0 and robot1 end-effector positions from state trajectories.
        
        Args:
            trajectories: List of trajectory arrays, each (T, state_dim)
            
        Returns:
            robot0_trajectories: List of robot0 EEF trajectories, each (T, 3)
            robot1_trajectories: List of robot1 EEF trajectories, each (T, 3)
        """
        robot0_trajectories = []
        robot1_trajectories = []
        
        for traj in trajectories:
            robot0_eef = traj[:, self.robot0_eef_pos_idx]  # (T, 3)
            robot1_eef = traj[:, self.robot1_eef_pos_idx]  # (T, 3)
            
            robot0_trajectories.append(robot0_eef)
            robot1_trajectories.append(robot1_eef)
        
        return robot0_trajectories, robot1_trajectories
    
    def create_robot_trajectory_plot(self,
                                   trajectories_dict: Dict[str, List[np.ndarray]],
                                   robot_name: str,
                                   robot_idx: slice,
                                   title: str,
                                   save_path: Optional[str] = None,
                                   max_trajs_per_policy: int = 1000) -> go.Figure:
        """
        Create a 3D trajectory plot for one robot's end-effector.
        
        Args:
            trajectories_dict: Dict mapping policy names to lists of trajectories
            robot_name: Name for the robot (e.g., "Robot0", "Robot1")
            robot_idx: Slice for extracting robot EEF positions
            title: Plot title
            save_path: Optional path to save the plot
            max_trajs_per_policy: Maximum number of trajectories to show per policy
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Colors for different policies - muted, professional colors
        color_map = {
            "Pretrained": "rgba(107, 142, 35, 0.6)",   # Muted navy blue
            "Current": "rgba(255, 127, 80, 0.6)",      # Coral - visible but not too bright
            "Expert": "rgba(70, 102, 153, 0.6)",       # Olive drab - muted green
        }
        
        for policy_name, trajectories in trajectories_dict.items():
            # Determine color and display name
            if "Step" in policy_name:
                color = color_map.get("Current", "rgba(255, 127, 80, 0.6)")
                display_name = "Finetuned"
            elif policy_name == "Expert":
                color = color_map.get("Expert", "rgba(70, 102, 153, 0.6)")
                display_name = "Expert"
            else:
                color = color_map.get(policy_name, "rgba(107, 142, 35, 0.6)")
                display_name = policy_name
            
            # Plot trajectories for this policy
            for i, traj in enumerate(trajectories[:max_trajs_per_policy]):
                # Extract robot EEF positions
                robot_eef = traj[:, robot_idx]  # (T, 3)
                
                # Only show legend for first trajectory of each policy
                show_legend = (i == 0)
                
                # Plot trajectory as connected line with small dots for each state
                fig.add_trace(go.Scatter3d(
                    x=robot_eef[:, 0],
                    y=robot_eef[:, 1], 
                    z=robot_eef[:, 2],
                    mode='lines+markers',
                    name=display_name if show_legend else "",
                    showlegend=show_legend,
                    line=dict(color=color, width=2),
                    marker=dict(size=2, color=color),  # Small dots for each state
                    hovertemplate=f'<b>{policy_name} Traj {i}</b><br>' +
                                  'X: %{x:.3f}<br>' +
                                  'Y: %{y:.3f}<br>' +
                                  'Z: %{z:.3f}<br>' +
                                  'Step: %{pointNumber}<extra></extra>'
                ))
        
        # Update layout
        fig.update_layout(
            title=f"{title} - {robot_name} End-Effector Trajectories",
            scene=dict(
                xaxis_title="X Position",
                yaxis_title="Y Position", 
                zaxis_title="Z Position",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
                aspectmode='cube'  # Equal aspect ratio for realistic 3D view
            ),
            showlegend=True,
            height=700,
            hovermode='closest'
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_comparison_plot(self,
                              trajectories_dict: Dict[str, List[np.ndarray]],
                              title: str = "Robot Trajectory Comparison",
                              save_path: Optional[str] = None,
                              max_trajs_per_policy: int = 1000) -> Tuple[go.Figure, go.Figure]:
        """
        Create comparison plots for both robots' trajectories.
        
        Args:
            trajectories_dict: Dict mapping policy names to lists of trajectories
            title: Base title for plots
            save_path: Optional base path to save plots (will add _robot0, _robot1)
            max_trajs_per_policy: Maximum number of trajectories to show per policy
            
        Returns:
            Tuple of (robot0_figure, robot1_figure)
        """
        # Create robot0 plot
        robot0_save_path = save_path.replace('.html', '_robot0.html') if save_path else None
        robot0_fig = self.create_robot_trajectory_plot(
            trajectories_dict=trajectories_dict,
            robot_name="Robot0",
            robot_idx=self.robot0_eef_pos_idx,
            title=title,
            save_path=robot0_save_path,
            max_trajs_per_policy=max_trajs_per_policy
        )
        
        # Create robot1 plot
        robot1_save_path = save_path.replace('.html', '_robot1.html') if save_path else None
        robot1_fig = self.create_robot_trajectory_plot(
            trajectories_dict=trajectories_dict,
            robot_name="Robot1", 
            robot_idx=self.robot1_eef_pos_idx,
            title=title,
            save_path=robot1_save_path,
            max_trajs_per_policy=max_trajs_per_policy
        )
        
        return robot0_fig, robot1_fig


# Legacy function for compatibility (not used with new approach)
def collect_evaluation_trajectories(env, 
                                   get_action_fn,
                                   n_episodes: int = 5,
                                   fixed_seeds: Optional[List[int]] = None,
                                   max_steps: int = 1000) -> Tuple[List[np.ndarray], List[float], List[bool]]:
    """
    Collect trajectories from evaluation episodes.
    
    Args:
        env: Environment (vectorized)
        get_action_fn: Function to get actions from observations
        n_episodes: Number of episodes per environment
        fixed_seeds: Optional list of seeds for deterministic reset
        max_steps: Maximum steps per episode
        
    Returns:
        trajectories: List of state trajectories
        rewards: List of episode rewards
        successes: List of success flags
    """
    n_envs = env.num_envs if hasattr(env, 'num_envs') else 1
    
    trajectories = []
    episode_rewards = []
    episode_successes = []
    
    for episode in range(n_episodes):
        # Reset with fixed seeds if provided
        options_venv = [{} for _ in range(n_envs)]
        if fixed_seeds is not None:
            for i in range(n_envs):
                seed_idx = (episode * n_envs + i) % len(fixed_seeds)
                options_venv[i]['seed'] = fixed_seeds[seed_idx] + 10000  # Offset for eval
        
        # Reset environments
        obs = env.reset(options=options_venv) if hasattr(env, 'reset') else env.reset()
        
        # Collect trajectories for each environment
        env_trajectories = [[] for _ in range(n_envs)]
        env_rewards = np.zeros(n_envs)
        env_done = np.zeros(n_envs, dtype=bool)
        
        for step in range(max_steps):
            # Store observations
            if isinstance(obs, dict):
                states = obs['state']  # (n_envs, obs_dim) or (n_envs, cond_steps, obs_dim)
                if states.ndim == 3:
                    states = states[:, -1]  # Take last observation in sequence
            else:
                states = obs
            
            for i in range(n_envs):
                if not env_done[i]:
                    env_trajectories[i].append(states[i].copy())
            
            # Get actions
            with torch.no_grad():
                actions = get_action_fn(obs)
            
            # Step environment
            obs, rewards, dones, infos = env.step(actions)
            
            # Update rewards and done flags
            for i in range(n_envs):
                if not env_done[i]:
                    env_rewards[i] += rewards[i]
                    if dones[i]:
                        env_done[i] = True
            
            # Check if all done
            if np.all(env_done):
                break
        
        # Store results
        for i in range(n_envs):
            if len(env_trajectories[i]) > 0:
                trajectories.append(np.array(env_trajectories[i]))
                episode_rewards.append(env_rewards[i])
                episode_successes.append(env_rewards[i] >= 1.0)  # Assuming binary success
    
    return trajectories, episode_rewards, episode_successes