"""
Replay buffer for off-policy reinforcement learning.

This module provides a replay buffer implementation that can store
transitions for off-policy RL algorithms like SAC, TD3, etc.

"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
import logging

log = logging.getLogger(__name__)


class ReplayBuffer:
    """
    A simple replay buffer for storing transitions.
    
    Stores transitions as (state, noise, action, reward, next_state, done)
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        noise_dim: int,
        max_size: int = 1000000,
        n_envs: int = 1,  # Number of parallel environments
        cond_steps: int = 1,  # Number of observation history steps
        horizon_steps: int = 8,  # Number of action horizon steps
        device: str = "cuda",
        gamma: float = 0.99,  # Discount factor for MC return computation
        log_q_overestimation: bool = False,  # Whether to compute MC returns
        # RLPD settings
        use_rlpd: bool = False,
        expert_ratio: float = 0.5,  # Ratio of expert data in each batch
        expert_dataset = None,  # Expert dataset for RLPD sampling
        # N-step returns
        use_n_step: bool = False,  # Whether to use n-step returns
        n_step: int = 1,  # Number of steps for n-step returns
        # Expert dataset n-step settings (can differ from online data)
        expert_use_n_step: bool = False,  # Whether expert data uses n-step returns
        expert_n_step: int = 1,  # Expert data n-step value
    ):
        """
        Initialize replay buffer for vectorized environments.
        
        Args:
            obs_dim: int - observation dimension
            action_dim: int - action dimension
            noise_dim: int - noise dimension
            max_size: int - maximum number of timesteps to store
            n_envs: int - number of parallel environments
            cond_steps: int - number of observation history steps (for chunked data)
            horizon_steps: int - number of action horizon steps
            device: str - device to store tensors on
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.noise_dim = noise_dim
        self.n_envs = n_envs
        self.cond_steps = cond_steps
        self.horizon_steps = horizon_steps
        self.max_size = max_size
        self.device = device
        self.gamma = gamma
        self.log_q_overestimation = log_q_overestimation
        self.use_n_step = use_n_step
        self.n_step = n_step
        
        # RLPD settings
        self.use_rlpd = use_rlpd
        self.expert_ratio = expert_ratio
        self.expert_dataset = expert_dataset
        
        # Expert dataset n-step settings
        self.expert_use_n_step = expert_use_n_step
        self.expert_n_step = expert_n_step
        if self.use_rlpd and self.expert_dataset is not None:
            log.info(f"RLPD enabled with expert_ratio={expert_ratio}")
            log.info(f"Expert dataset size: {len(expert_dataset)}")
        else:
            log.info("Regular replay buffer (RLPD disabled)")
        
        # Initialize storage for vectorized environment data
        # Shape: (max_timesteps, n_envs, ...)  
        # Note: max_size refers to number of timesteps, total transitions = max_size * n_envs
        self.states = torch.zeros((max_size, n_envs, cond_steps, obs_dim), dtype=torch.float32, device=device)
        self.noises = torch.zeros((max_size, n_envs, horizon_steps, action_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((max_size, n_envs, horizon_steps, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((max_size, n_envs, 1), dtype=torch.float32, device=device)
        self.next_states = torch.zeros((max_size, n_envs, cond_steps, obs_dim), dtype=torch.float32, device=device)
        self.dones = torch.zeros((max_size, n_envs, 1), dtype=torch.bool, device=device)
        
        # Only allocate MC return buffer if needed
        if self.log_q_overestimation:
            self.mc_return = torch.zeros((max_size, n_envs, 1), dtype=torch.float32, device=device)
            # Episode tracking for MC return computation
            self.episode_rewards = [[] for _ in range(n_envs)]
            self.episode_indices = [[] for _ in range(n_envs)]
        
        # Buffer state
        self.ptr = 0
        self.size = 0
        
        log.info(f"ReplayBuffer initialized with max_size={max_size}")
    
    def add(
        self,
        state: torch.Tensor,
        noise: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
    ):
        """
        Add vectorized environment transitions to the replay buffer.
        
        Expects data from n_envs parallel environments at a single timestep.
        
        Args:
            state: (n_envs, cond_steps, obs_dim) - chunked current states from all envs
            noise: (n_envs, horizon_steps, action_dim) - chunked noise from all envs
            action: (n_envs, horizon_steps, action_dim) - chunked actions from all envs
            reward: (n_envs, 1) - rewards from all envs
            next_state: (n_envs, cond_steps, obs_dim) - chunked next states from all envs
            done: (n_envs, 1) - done flags from all envs
        """
        batch_size = state.shape[0]  # should equal n_envs
        # assert batch_size == self.n_envs, f"Expected {self.n_envs} environments, got {batch_size}"
        
        # Store entire timestep across all environments
        self.states[self.ptr] = state  # (n_envs, cond_steps, obs_dim)
        self.noises[self.ptr] = noise  # (n_envs, horizon_steps, action_dim)
        self.actions[self.ptr] = action  # (n_envs, horizon_steps, action_dim)
        self.rewards[self.ptr] = reward  # (n_envs, 1)
        self.next_states[self.ptr] = next_state  # (n_envs, cond_steps, obs_dim)
        self.dones[self.ptr] = done  # (n_envs, 1)
        
        # Track episode data for MC return computation only if needed
        if self.log_q_overestimation:
            for env_idx in range(self.n_envs):
                # Add current transition to episode
                self.episode_rewards[env_idx].append(reward[env_idx].item())
                self.episode_indices[env_idx].append((self.ptr, env_idx))
                
                # If episode is done, compute MC returns for the entire episode
                if done[env_idx].item():
                    # Compute discounted returns for this episode
                    episode_rewards = self.episode_rewards[env_idx]
                    episode_indices = self.episode_indices[env_idx]
                    
                    # Backward pass to compute MC returns
                    mc_return = 0.0
                    for i in reversed(range(len(episode_rewards))):
                        mc_return = episode_rewards[i] + self.gamma * mc_return
                        ptr_idx, env_idx_check = episode_indices[i]
                        assert env_idx_check == env_idx
                        self.mc_return[ptr_idx, env_idx, 0] = mc_return
                    
                    # Clear episode data for this environment
                    self.episode_rewards[env_idx] = []
                    self.episode_indices[env_idx] = []
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size: int, expert_ratio: Optional[float] = None) -> Tuple[torch.Tensor, ...]:
        """
        Sample a batch of transitions from the replay buffer.
        
        If RLPD is enabled, uses symmetric sampling (expert_ratio from expert data,
        remaining from online data). Otherwise uses standard uniform sampling.
        
        Args:
            batch_size: int - number of transitions to sample
            expert_ratio: Optional[float] - override the default expert_ratio for adaptive scheduling
            
        Returns:
            tuple: (state, noise, action, reward, next_state, done, mc_return)
                state: (batch_size, cond_steps, obs_dim)
                noise: (batch_size, horizon_steps, action_dim)
                action: (batch_size, horizon_steps, action_dim)
                reward: (batch_size, 1)
                next_state: (batch_size, cond_steps, obs_dim)
                done: (batch_size, 1)
                mc_return: (batch_size, 1)
        """
        if self.use_rlpd and self.expert_dataset is not None:
            return self._sample_rlpd(batch_size, expert_ratio=expert_ratio)
        else:
            return self._sample_standard(batch_size)
    
    def _sample_standard(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Standard uniform sampling from online replay buffer."""
        if self.size == 0:
            raise ValueError("Cannot sample from empty buffer")
        
        if self.use_n_step:
            # For n-step returns, ensure we don't sample from the last n-1 transitions
            safe_size = max(1, self.size - self.n_step + 1)
            total_transitions = safe_size * self.n_envs
        else:
            # Total number of transitions available (timesteps * environments)
            total_transitions = self.size * self.n_envs
        
        # Sample random flat indices from all available transitions
        flat_indices = np.random.randint(0, total_transitions, size=batch_size)
        
        # Convert flat indices back to (timestep, env) indices
        timestep_indices = flat_indices // self.n_envs
        env_indices = flat_indices % self.n_envs
        
        if self.use_n_step:
            # Use fixed n-step for all samples
            n_steps_per_sample = np.full(batch_size, self.n_step)
            n_steps_tensor = torch.tensor(n_steps_per_sample, device=self.device, dtype=torch.long).unsqueeze(1)  # (batch_size, 1)

            # Data source labels: 0 for online data
            data_source = torch.zeros(batch_size, 1, device=self.device, dtype=torch.long)

            # Compute n-step returns
            n_step_rewards = torch.zeros(batch_size, 1, device=self.device)
            n_step_dones = torch.zeros(batch_size, 1, dtype=torch.bool, device=self.device)
            
            for step in range(self.n_step):
                step_indices = (timestep_indices + step) % self.max_size
                step_rewards = self.rewards[step_indices, env_indices]  # (batch_size, 1)
                step_dones = self.dones[step_indices, env_indices]  # (batch_size, 1)
                
                # Add discounted reward only if episode hasn't ended
                n_step_rewards += (self.gamma ** step) * step_rewards * (~n_step_dones).float()
                # Mark as done if any step in the n-step window is done
                n_step_dones = n_step_dones | step_dones
            
            # Get the state n steps ahead for bootstrapping
            n_step_indices = (timestep_indices + self.n_step) % self.max_size
            n_step_next_states = self.next_states[n_step_indices - 1, env_indices]  # -1 because we want the next_state of the (n-1)th transition
            
            if self.log_q_overestimation:
                return (
                    self.states[timestep_indices, env_indices],  # (batch_size, cond_steps, obs_dim)
                    self.noises[timestep_indices, env_indices],   # (batch_size, horizon_steps, action_dim)
                    self.actions[timestep_indices, env_indices],  # (batch_size, horizon_steps, action_dim)
                    n_step_rewards,  # (batch_size, 1) - n-step discounted rewards
                    n_step_next_states,  # (batch_size, cond_steps, obs_dim) - state n steps ahead
                    n_step_dones,  # (batch_size, 1) - done if any step is done
                    self.mc_return[timestep_indices, env_indices],   # (batch_size, 1)
                    n_steps_tensor,  # (batch_size, 1) - actual n-step used for each sample
                    data_source,  # (batch_size, 1) - 0 for online data
                )
            else:
                return (
                    self.states[timestep_indices, env_indices],  # (batch_size, cond_steps, obs_dim)
                    self.noises[timestep_indices, env_indices],   # (batch_size, horizon_steps, action_dim)
                    self.actions[timestep_indices, env_indices],  # (batch_size, horizon_steps, action_dim)
                    n_step_rewards,  # (batch_size, 1) - n-step discounted rewards
                    n_step_next_states,  # (batch_size, cond_steps, obs_dim) - state n steps ahead
                    n_step_dones,  # (batch_size, 1) - done if any step is done
                    n_steps_tensor,  # (batch_size, 1) - actual n-step used for each sample
                    data_source,  # (batch_size, 1) - 0 for online data
                )
        else:
            # Original single-step returns - default n_step=1 for all samples
            n_steps_tensor = torch.ones(batch_size, 1, device=self.device, dtype=torch.long)
            
            # Data source labels: 0 for online data
            data_source = torch.zeros(batch_size, 1, device=self.device, dtype=torch.long)
            
            if self.log_q_overestimation:
                return (
                    self.states[timestep_indices, env_indices],  # (batch_size, cond_steps, obs_dim)
                    self.noises[timestep_indices, env_indices],   # (batch_size, horizon_steps, action_dim)
                    self.actions[timestep_indices, env_indices],  # (batch_size, horizon_steps, action_dim)
                    self.rewards[timestep_indices, env_indices],      # (batch_size, 1)
                    self.next_states[timestep_indices, env_indices], # (batch_size, cond_steps, obs_dim)
                    self.dones[timestep_indices, env_indices],       # (batch_size, 1)
                    self.mc_return[timestep_indices, env_indices],   # (batch_size, 1)
                    n_steps_tensor,  # (batch_size, 1) - n_step=1 for single-step returns
                    data_source,  # (batch_size, 1) - 0 for online data
                )
            else:
                return (
                    self.states[timestep_indices, env_indices],  # (batch_size, cond_steps, obs_dim)
                    self.noises[timestep_indices, env_indices],   # (batch_size, horizon_steps, action_dim)
                    self.actions[timestep_indices, env_indices],  # (batch_size, horizon_steps, action_dim)
                    self.rewards[timestep_indices, env_indices],      # (batch_size, 1)
                    self.next_states[timestep_indices, env_indices], # (batch_size, cond_steps, obs_dim)
                    self.dones[timestep_indices, env_indices],       # (batch_size, 1)
                    n_steps_tensor,  # (batch_size, 1) - n_step=1 for single-step returns
                    data_source,  # (batch_size, 1) - 0 for online data
                )
    
    def _sample_rlpd(self, batch_size: int, expert_ratio: Optional[float] = None) -> Tuple[torch.Tensor, ...]:
        """RLPD symmetric sampling: mix expert data with online data."""
        # Use provided expert_ratio or fall back to default
        current_expert_ratio = expert_ratio if expert_ratio is not None else self.expert_ratio
        
        # Calculate split
        expert_batch_size = int(batch_size * current_expert_ratio)
        online_batch_size = batch_size - expert_batch_size
        
        # Sample expert data
        expert_transitions = self._sample_expert_data(expert_batch_size)
        
        # Sample online data (if available)
        if self.size > 0 and online_batch_size > 0:
            online_transitions = self._sample_standard(online_batch_size)
            # Concatenate expert and online data
            combined_transitions = tuple(
                torch.cat([expert_batch, online_batch], dim=0)
                for expert_batch, online_batch in zip(expert_transitions, online_transitions)
            )
        else:
            # Only expert data available
            combined_transitions = expert_transitions
        
        return combined_transitions
    
    def _sample_expert_data(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample from expert dataset and convert to same format as replay buffer."""
        # Sample random indices from expert dataset
        expert_indices = np.random.randint(0, len(self.expert_dataset), size=batch_size)
        
        # Get expert transitions
        expert_batch = [self.expert_dataset[idx] for idx in expert_indices]
        
        # Convert to tensors and proper format
        states = torch.stack([torch.tensor(transition.conditions['state'], dtype=torch.float32, device=self.device) 
                             for transition in expert_batch])
        actions = torch.stack([torch.tensor(transition.actions, dtype=torch.float32, device=self.device) 
                              for transition in expert_batch]) 
        next_states = torch.stack([torch.tensor(transition.conditions['next_state'], dtype=torch.float32, device=self.device) 
                                  for transition in expert_batch])
        rewards = torch.stack([torch.tensor(transition.rewards, dtype=torch.float32, device=self.device) 
                              for transition in expert_batch])
        dones = torch.stack([torch.tensor(transition.dones, dtype=torch.bool, device=self.device) 
                            for transition in expert_batch])
        
        # Generate random noise for expert transitions (needed for consistency)
        noises = torch.randn(batch_size, self.horizon_steps, self.action_dim, device=self.device)
        
        # Expert data uses its configured n_step value (can differ from online data)
        # The expert dataset handles n-step computation in its __getitem__ method
        # The model will correctly apply gamma^expert_n_step for these transitions
        n_steps_tensor = torch.ones(batch_size, 1, device=self.device, dtype=torch.long) * self.expert_n_step
        
        # Data source labels: 1 for expert data
        data_source = torch.ones(batch_size, 1, device=self.device, dtype=torch.long)
        
        if self.log_q_overestimation:
            # Get MC returns for expert data
            mc_returns = torch.stack([torch.tensor(transition.mc_return, dtype=torch.float32, device=self.device) 
                                     for transition in expert_batch])
            return states, noises, actions, rewards, next_states, dones, mc_returns, n_steps_tensor, data_source
        else:
            return states, noises, actions, rewards, next_states, dones, n_steps_tensor, data_source
    
    
    def __len__(self) -> int:
        """Return the number of timesteps stored in the buffer."""
        return self.size
    
    def get_total_transitions(self) -> int:
        """Return the total number of individual transitions (timesteps * environments)."""
        return self.size * self.n_envs
    
    def is_full(self) -> bool:
        """Check if the buffer is full."""
        return self.size == self.max_size
    
    def clear(self):
        """Clear the replay buffer."""
        self.ptr = 0
        self.size = 0
        log.info("Replay buffer cleared")


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized replay buffer that samples transitions based on their priority.
    
    This implementation uses a simple priority scheme based on TD-error.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        noise_dim: int,
        max_size: int = 1000000,
        alpha: float = 0.6,
        beta: float = 0.4,
        device: str = "cuda",
    ):
        """
        Initialize prioritized replay buffer.
        
        Args:
            obs_dim: int - observation dimension
            action_dim: int - action dimension
            noise_dim: int - noise dimension
            max_size: int - maximum number of transitions to store
            alpha: float - priority exponent
            beta: float - importance sampling exponent
            device: str - device to store tensors on
        """
        super().__init__(obs_dim, action_dim, noise_dim, max_size, 1, 8, device)  # cond_steps=1, horizon_steps=8 for prioritized buffer
        
        self.alpha = alpha
        self.beta = beta
        
        # Priority storage
        self.priorities = torch.zeros((max_size,), dtype=torch.float32, device=device)
        self.max_priority = 1.0
        
        log.info(f"PrioritizedReplayBuffer initialized with alpha={alpha}, beta={beta}")
    
    def add(
        self,
        state: torch.Tensor,
        noise: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
        priority: Optional[torch.Tensor] = None,
    ):
        """
        Add a transition to the prioritized replay buffer.
        
        Args:
            state: (B, cond_steps, obs_dim) - chunked current state
            noise: (B, horizon_steps, action_dim) - chunked noise
            action: (B, horizon_steps, action_dim) - chunked action
            reward: (B, 1) - reward
            next_state: (B, cond_steps, obs_dim) - chunked next state
            done: (B, 1) - done flag
            priority: (B,) - priority values (optional)
        """
        batch_size = state.shape[0]
        
        if priority is None:
            priority = torch.full((batch_size,), self.max_priority, device=self.device)
        
        # Handle single transition
        if batch_size == 1:
            super().add(state, noise, action, reward, next_state, done)
            self.priorities[self.ptr - 1] = priority.squeeze(0)
        else:
            # Handle batch of transitions
            for i in range(batch_size):
                super().add(
                    state[i:i+1], noise[i:i+1], action[i:i+1],
                    reward[i:i+1], next_state[i:i+1], done[i:i+1]
                )
                self.priorities[self.ptr - 1] = priority[i]
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample a batch of transitions based on priority.
        
        Args:
            batch_size: int - number of transitions to sample
            
        Returns:
            tuple: (state, noise, action, reward, next_state, done, weights, indices)
                state: (batch_size, cond_steps, obs_dim)
                noise: (batch_size, horizon_steps, action_dim)
                action: (batch_size, horizon_steps, action_dim)
                reward: (batch_size, 1)
                next_state: (batch_size, cond_steps, obs_dim)
                done: (batch_size, 1)
                weights: (batch_size,) - importance sampling weights
                indices: (batch_size,) - sampled indices
        """
        if self.size < batch_size:
            raise ValueError(f"Cannot sample {batch_size} transitions from buffer with {self.size} transitions")
        
        # Compute sampling probabilities
        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(self.size, size=batch_size, p=probs.cpu().numpy())
        
        # Compute importance sampling weights
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        return (
            self.states[indices],
            self.noises[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            weights,
            indices,
        )
    
    def update_priorities(self, indices: np.ndarray, priorities: torch.Tensor):
        """
        Update priorities for given indices.
        
        Args:
            indices: (B,) - indices to update
            priorities: (B,) - new priority values
        """
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, priorities.max().item()) 