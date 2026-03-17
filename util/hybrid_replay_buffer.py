"""
Hybrid replay buffer for intermediate chunk sampling.

This buffer combines episode storage with efficient chunk conversion for 
fast vectorized sampling while respecting episode boundaries.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
import logging
from torch.utils.data import WeightedRandomSampler, DataLoader, Sampler

log = logging.getLogger(__name__)


class HybridReplayBuffer:
    """
    Hybrid replay buffer that combines episode storage with efficient chunk conversion.
    
    Storage strategy:
    1. Collect complete episodes from parallel environments in simple list format
    2. Convert completed episodes to valid training chunks for fast sampling
    3. Maintain both episode storage (for ongoing episodes) and chunk storage (for sampling)
    
    This approach balances correctness (respecting episode boundaries) with performance
    (fast vectorized sampling without boundary checks).
    
    Key features:
    - Stores full trajectory data: o_0, a_0, r_0, o_1, a_1, r_1, ..., o_T
    - Converts episodes to all valid chunks up to T-horizon_steps
    - Provides fast vectorized sampling without runtime boundary checks
    - Properly handles varying episode lengths and early termination
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        noise_dim: int,
        max_size: int = 1000000,
        n_envs: int = 1,
        cond_steps: int = 1,
        horizon_steps: int = 8,
        device: str = "cuda",
        gamma: float = 0.99,
        log_q_overestimation: bool = False,
        # RLPD settings
        use_rlpd: bool = False,
        expert_ratio: float = 0.5,
        expert_dataset = None,
        # N-step returns
        use_n_step: bool = False,
        n_step: int = 1,
        # Expert dataset n-step settings
        expert_use_n_step: bool = False,
        expert_n_step: int = 1,
    ):
        """
        Initialize hybrid replay buffer with episode and chunk storage.
        
        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension  
            noise_dim: Noise dimension (for generated noise during sampling)
            max_size: Maximum number of chunks to store (not timesteps)
            n_envs: Number of parallel environments
            cond_steps: Number of observation history steps
            horizon_steps: Action horizon for chunking (e.g., 8)
            device: Device for tensor storage
            gamma: Discount factor for MC return computation (NOT for within-chunk rewards)
            log_q_overestimation: Whether to compute MC returns
            use_rlpd: Whether to use RLPD sampling
            expert_ratio: Ratio of expert data in each batch
            expert_dataset: Expert dataset for RLPD
            use_n_step: Whether to use n-step returns
            n_step: Number of steps for n-step returns
            expert_use_n_step: Whether expert data uses n-step returns
            expert_n_step: Expert data n-step value
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
        
        # Episode storage: List of episodes for each environment
        # Each episode is a dict with 'observations', 'actions', 'rewards'
        self.ongoing_episodes = [None for _ in range(n_envs)]
        
        # Chunk storage for fast sampling
        # Shape: (max_size, ...) - stores preprocessed chunks ready for training
        # Each chunk represents a valid transition starting at some timestep t
        self.chunk_states = torch.zeros((max_size, cond_steps, obs_dim), 
                                       dtype=torch.float32, device=device)
        self.chunk_actions = torch.zeros((max_size, horizon_steps, action_dim), 
                                        dtype=torch.float32, device=device)
        self.chunk_rewards = torch.zeros((max_size, 1), 
                                        dtype=torch.float32, device=device)
        self.chunk_next_states = torch.zeros((max_size, cond_steps, obs_dim), 
                                            dtype=torch.float32, device=device)
        self.chunk_dones = torch.zeros((max_size, 1), 
                                      dtype=torch.bool, device=device)
        
        # MC return storage if needed
        if self.log_q_overestimation:
            self.chunk_mc_returns = torch.zeros((max_size, 1), 
                                              dtype=torch.float32, device=device)
        
        # Buffer state
        self.ptr = 0  # Points to next chunk storage location
        self.size = 0  # Number of chunks stored
        self.num_episodes = 0  # Number of complete episodes processed
        
        log.info(f"HybridReplayBuffer initialized")
        log.info(f"  Max chunks: {max_size}")
        log.info(f"  Horizon steps: {horizon_steps}")
        log.info(f"  Cond steps: {cond_steps}")
        log.info(f"  Parallel envs: {n_envs}")
        if use_rlpd and expert_dataset:
            log.info(f"  RLPD enabled with expert_ratio={expert_ratio}")
            log.info(f"  Expert dataset size: {len(expert_dataset)}")
    
    def add(
        self,
        state: torch.Tensor,
        noise: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
        full_trajectory_info: List[Dict[str, Any]] = None,
    ):
        """
        Add transitions from full trajectory data.
        
        This method processes the full trajectory information from MultiStepFull wrapper,
        accumulating episode data and converting complete episodes to chunks.
        
        Args:
            state: (n_envs, cond_steps, obs_dim) - State at chunk start (for compatibility)
            noise: (n_envs, horizon_steps, action_dim) - Noise (not used, will generate during sampling)
            action: (n_envs, horizon_steps, action_dim) - Chunked actions (for compatibility)
            reward: (n_envs, 1) - Accumulated reward (for compatibility)
            next_state: (n_envs, cond_steps, obs_dim) - State at chunk end (for compatibility)
            done: (n_envs, 1) - Done flag
            full_trajectory_info: List[Dict] with full trajectory data from MultiStepFull wrapper
                Each dict contains:
                - 'initial_obs': o_0 if this is the first chunk, else None
                - 'observations': List of observations [o_1, ..., o_k] where k <= horizon_steps
                - 'actions': List of actions [a_0, ..., a_{k-1}]
                - 'rewards': List of rewards [r_0, ..., r_{k-1}]
                - 'include_initial': Whether initial_obs is included
        """
        if full_trajectory_info is None:
            # Fallback to standard chunked storage if no full trajectory data
            log.warning("No full trajectory info provided, skipping")
            return
        
        # Process each environment's trajectory data
        for env_idx in range(self.n_envs):
            if env_idx >= len(full_trajectory_info):
                continue
                
            info = full_trajectory_info[env_idx]
            if not info or 'observations' not in info:
                raise Exception("no observation")
                continue
            
            # Extract trajectory data
            initial_obs = info.get('initial_obs', None)
            observations = info['observations']  # Length k: [o_1, ..., o_k]
            actions = info['actions']           # Length k: [a_0, ..., a_{k-1}]
            rewards = info['rewards']           # Length k: [r_0, ..., r_{k-1}]
            dones = info['dones']               # Length k: [done_0, ..., done_{k-1}]
            include_initial = info.get('include_initial', False)
            
            # Initialize or update ongoing episode for this environment
            if self.ongoing_episodes[env_idx] is None or include_initial:
                # Start new episode
                self.ongoing_episodes[env_idx] = {
                    'observations': [],  # Will store [o_0, o_1, ..., o_T]
                    'actions': [],       # Will store [a_0, a_1, ..., a_{T-1}]
                    'rewards': [],       # Will store [r_0, r_1, ..., r_{T-1}]
                    'dones': [],         # Will store [done_0, done_1, ..., done_{T-1}]
                }
                # Add initial observation o_0
                if initial_obs is not None:
                    self.ongoing_episodes[env_idx]['observations'].append(
                        self._to_tensor(initial_obs)
                    )
            
            episode = self.ongoing_episodes[env_idx]
            
            # Add timestep data to episode
            # Note: observations list will be one longer than actions/rewards/dones
            num_steps = len(actions)  # Number of steps actually executed
            for i in range(num_steps):
                episode['actions'].append(self._to_tensor(actions[i]))
                episode['rewards'].append(rewards[i])
                episode['dones'].append(dones[i])
                episode['observations'].append(self._to_tensor(observations[i]))
            
            # Check if episode is done
            # The done tensor has shape (n_envs, 1) from agent's unsqueeze(-1)
            done_from_agent = done[env_idx, 0].item()
            done_from_trajectory = dones[-1] if dones else False
            
            # These should be equal if implementation is correct
            if done_from_agent != done_from_trajectory:
                log.warning(f"Done flag mismatch for env {env_idx}: agent={done_from_agent}, trajectory={done_from_trajectory}")
                
            if done_from_agent:
                # Convert complete episode to chunks and store
                self._process_complete_episode(episode)
                # Clear episode storage
                self.ongoing_episodes[env_idx] = None
    
    def _to_tensor(self, data):
        """Convert data to tensor if needed."""
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        elif isinstance(data, dict):
            # For dict observations, recursively convert each value
            return {k: self._to_tensor(v) for k, v in data.items()}
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data).float().to(self.device)
        else:
            return torch.tensor(data).float().to(self.device)
    
    def _process_complete_episode(self, episode: Dict[str, List]):
        """
        Convert a complete episode to training chunks and store them.
        
        For an episode with T timesteps, creates chunks starting at timesteps
        that ensure we have enough observation history and don't go past episode end.
        
        Each chunk contains:
        - State at t (with history if cond_steps > 1)
        - Actions from t to t+horizon_steps-1
        - Sum of rewards from t to t+horizon_steps-1 (NO discounting within chunk)
        - State at t+horizon_steps (with history)
        - Done flag (true if episode ends within chunk)
        
        Args:
            episode: Dict with 'observations' [o_0, ..., o_T], 
                    'actions' [a_0, ..., a_{T-1}], 'rewards' [r_0, ..., r_{T-1}],
                    'dones' [done_0, ..., done_{T-1}]
        """
        observations = episode['observations']  # Length T+1: [o_0, o_1, ..., o_T]
        actions = episode['actions']           # Length T: [a_0, a_1, ..., a_{T-1}]
        rewards = episode['rewards']           # Length T: [r_0, r_1, ..., r_{T-1}]
        dones = episode['dones']               # Length T: [done_0, ..., done_{T-1}]
        
        T = len(actions)  # Number of transitions in episode
        
        # Need at least horizon_steps transitions to create a chunk
        if T < self.horizon_steps:
            # Episode too short for any chunks
            return 0
        
        # Determine valid range for chunk starts based on cond_steps
        # We need at least cond_steps observations before start_t to build history
        min_start_t = max(0, self.cond_steps - 1)  # Need history for s_t
        
        # Also need enough observations for next_state history at t+horizon_steps
        # If we start at t, next state is at t+horizon_steps, which needs history back to t+horizon_steps-cond_steps+1
        # So we need t+horizon_steps < len(observations) = T+1
        max_start_t = T - self.horizon_steps + 1
        
        # Skip if no valid chunks can be created
        if min_start_t >= max_start_t:
            return 0
        
        chunks_created = 0
        
        for start_t in range(min_start_t, max_start_t):
            # Build state at start_t with history
            # For cond_steps=1: just o_t
            # For cond_steps>1: [o_{t-c+1}, ..., o_t] - all from real data
            if self.cond_steps == 1:
                state_t = observations[start_t].unsqueeze(0)
            else:
                # All indices are valid since start_t >= cond_steps-1
                history = []
                for i in range(self.cond_steps):
                    hist_idx = start_t - self.cond_steps + 1 + i
                    history.append(observations[hist_idx])
                state_t = torch.stack(history)
            
            # Extract chunk of actions [a_t, ..., a_{t+h-1}]
            # We know the episode has at least horizon_steps more transitions
            chunk_actions = torch.stack([actions[start_t + i] for i in range(self.horizon_steps)])
            
            # Sum rewards (NO discounting within chunk)
            # Transition definition: s_t, a_t:t+h-1, s_t+h, sum_{t'=t}^{t+h-1} r_{t'}
            chunk_reward = sum(rewards[start_t:start_t + self.horizon_steps])
            
            # Build next state at t+horizon_steps
            next_t = start_t + self.horizon_steps
            if self.cond_steps == 1:
                next_state_t = observations[next_t].unsqueeze(0)
            else:
                # Build history for next state [o_{next_t-c+1}, ..., o_{next_t}]
                # All indices are valid since we checked boundaries
                history = []
                for i in range(self.cond_steps):
                    hist_idx = next_t - self.cond_steps + 1 + i
                    history.append(observations[hist_idx])
                next_state_t = torch.stack(history)
            
            # Check if episode ends within this chunk using actual done signals
            # Done is true if any step within the chunk has done=True
            chunk_done = any(dones[start_t:start_t + self.horizon_steps])
            
            # Store chunk
            idx = self.ptr % self.max_size
            self.chunk_states[idx] = state_t
            self.chunk_actions[idx] = chunk_actions
            self.chunk_rewards[idx, 0] = chunk_reward
            self.chunk_next_states[idx] = next_state_t
            self.chunk_dones[idx, 0] = chunk_done
            
            # Compute MC return if needed (with discounting at chunk level)
            if self.log_q_overestimation:
                # MC return should be computed as:
                # sum_{t'=t}^{t+h-1} r_{t'} + gamma^1 * sum_{t'=t+h}^{t+2h-1} r_{t'} + gamma^2 * sum_{t'=t+2h}^{t+3h-1} r_{t'} + ...
                # Following agent/dataset/sequence.py:263:274 implementation
                mc_return = 0.0
                t = start_t
                chunk_idx = 0
                
                while t < T:
                    # Sum rewards for this chunk
                    chunk_end = min(t + self.horizon_steps, T)
                    chunk_reward_sum = sum(rewards[t:chunk_end])
                    
                    # Add discounted chunk reward (gamma^chunk_idx, not gamma^(chunk_idx * horizon_steps))
                    mc_return += (self.gamma ** chunk_idx) * chunk_reward_sum
                    
                    # Move to next chunk
                    t = chunk_end
                    chunk_idx += 1
                
                self.chunk_mc_returns[idx, 0] = mc_return
            
            # Update buffer pointers
            self.ptr = (self.ptr + 1) % self.max_size
            self.size = min(self.size + 1, self.max_size)
            chunks_created += 1
        
        # Increment episode counter when we've processed a complete episode
        self.num_episodes += 1
        
        return chunks_created
    
    def sample(self, batch_size: int, expert_ratio: Optional[float] = None) -> Tuple[torch.Tensor, ...]:
        """
        Sample a batch of transitions for training.
        
        Returns pre-converted chunks for fast sampling without boundary checks.
        
        Args:
            batch_size: Number of transitions to sample
            expert_ratio: Optional override for expert data ratio (for RLPD)
            
        Returns:
            Tuple of tensors:
            - states: (batch_size, cond_steps, obs_dim)
            - noises: (batch_size, horizon_steps, action_dim) - generated random noise
            - actions: (batch_size, horizon_steps, action_dim)
            - rewards: (batch_size, 1) - sum of rewards (no discounting)
            - next_states: (batch_size, cond_steps, obs_dim)
            - dones: (batch_size, 1)
            - [mc_returns]: (batch_size, 1) - if log_q_overestimation=True
            - n_steps: (batch_size, 1) - number of steps (for n-step returns)
            - data_source: (batch_size, 1) - 0 for online, 1 for expert
        """
        if self.use_rlpd and self.expert_dataset is not None:
            return self._sample_rlpd(batch_size, expert_ratio)
        else:
            return self._sample_standard(batch_size)
    
    def _sample_standard(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample from online data only."""
        if self.size == 0:
            raise ValueError("Cannot sample from empty buffer")
        
        if self.use_n_step:
            # For n-step returns, we need to aggregate multiple chunks
            # Since each chunk already represents horizon_steps transitions,
            # n-step here means we look at n non-overlapping chunks ahead
            safe_size = max(1, self.size - (self.n_step - 1) * self.horizon_steps)
            indices = np.random.randint(0, safe_size, size=batch_size)
            
            # Compute n-step returns by looking at ALL chunks within n_step window
            # But discount based on when the reward actually occurs (at END of chunk)
            n_step_rewards = torch.zeros(batch_size, 1, device=self.device)
            n_step_dones = torch.zeros(batch_size, 1, dtype=torch.bool, device=self.device)
            
            # Look at all chunks within the n-step window
            # For n_step=5, we want chunks at distances 0, 1, ..., (n_step-1)*horizon_steps
            max_distance = (self.n_step - 1) * self.horizon_steps + 1
            
            for k in range(max_distance):
                step_indices = indices + k  # Look at every chunk, not just every horizon_steps
                
                # Check if these indices are valid (don't go past buffer size)
                # Convert to tensor for consistent operations
                step_indices_tensor = torch.from_numpy(step_indices).to(self.device)
                valid_mask = (step_indices_tensor < self.size) & (~n_step_dones.squeeze())  # Also check if not done
                if not valid_mask.any():
                    break  # No valid indices left or all episodes done
                
                # Get rewards and dones for valid indices
                step_rewards = torch.zeros(batch_size, 1, device=self.device)
                step_dones_curr = torch.zeros(batch_size, 1, dtype=torch.bool, device=self.device)
                
                if valid_mask.sum() > 0:
                    valid_idx = step_indices_tensor[valid_mask]
                    step_rewards[valid_mask] = self.chunk_rewards[valid_idx].view(-1, 1)
                    step_dones_curr[valid_mask] = self.chunk_dones[valid_idx].view(-1, 1)
                
                # Calculate discount based on when reward occurs (at END of chunk)
                # Reward occurs at the end of chunk index+k, which is at timestep (index+k + horizon_steps - 1)
                # Distance from starting timestep index to reward = k + horizon_steps - 1
                # Horizon window number = (k + horizon_steps - 1) // horizon_steps
                discount_power = (k + self.horizon_steps - 1) // self.horizon_steps
                discount = self.gamma ** discount_power
                
                # Add discounted reward only if episode hasn't ended
                n_step_rewards += discount * step_rewards * (~n_step_dones).float()
                
                # Mark as done if any chunk is done
                n_step_dones = n_step_dones | step_dones_curr

            # Get the state after the last chunk in the n-step sequence
            n_step_indices = indices + (self.n_step - 1) * self.horizon_steps  # Non-overlapping: t + (n-1)*horizon_steps
            n_step_next_states = self.chunk_next_states[n_step_indices]
            
            # Debug: Show reward components for max n-step reward
            max_reward = n_step_rewards.max().item()
            if max_reward > 15:
                high_idx = torch.argmax(n_step_rewards).item()
                orig_idx = indices[high_idx]
                print(f"Max n-step reward {max_reward:.2f} at buffer index {orig_idx}:")
                for step in range(self.n_step):
                    step_idx = orig_idx + step
                    if step_idx < self.size:
                        step_reward = self.chunk_rewards[step_idx].item()
                        discount = (self.gamma ** step)
                        print(f"  Step {step}: {step_reward:.2f} * {discount:.3f} = {step_reward * discount:.2f}")
                        
            # Assert n_step rewards should be < 15 (success_steps_for_termination=15)
            assert torch.all(n_step_rewards < 15), f"N-step rewards exceed 15: max={max_reward:.2f}"
            
            # Generate random noise
            noises = torch.randn(batch_size, self.horizon_steps, self.action_dim, device=self.device)
            
            # n_step and data source
            n_steps_tensor = torch.ones(batch_size, 1, device=self.device, dtype=torch.long) * self.n_step
            data_source = torch.zeros(batch_size, 1, device=self.device, dtype=torch.long)
            
            if self.log_q_overestimation:
                return (
                    self.chunk_states[indices],      # (batch_size, cond_steps, obs_dim)
                    noises,                          # (batch_size, horizon_steps, action_dim)
                    self.chunk_actions[indices],     # (batch_size, horizon_steps, action_dim)
                    n_step_rewards,                  # (batch_size, 1) - n-step discounted rewards
                    n_step_next_states,              # (batch_size, cond_steps, obs_dim) - state n chunks ahead
                    n_step_dones,                    # (batch_size, 1) - done if any chunk is done
                    self.chunk_mc_returns[indices],  # (batch_size, 1)
                    n_steps_tensor,                  # (batch_size, 1)
                    data_source,                     # (batch_size, 1)
                )
            else:
                return (
                    self.chunk_states[indices],      # (batch_size, cond_steps, obs_dim)
                    noises,                          # (batch_size, horizon_steps, action_dim)
                    self.chunk_actions[indices],     # (batch_size, horizon_steps, action_dim)
                    n_step_rewards,                  # (batch_size, 1) - n-step discounted rewards
                    n_step_next_states,              # (batch_size, cond_steps, obs_dim) - state n chunks ahead
                    n_step_dones,                    # (batch_size, 1) - done if any chunk is done
                    n_steps_tensor,                  # (batch_size, 1)
                    data_source,                     # (batch_size, 1)
                )
        else:
            # Standard single-chunk sampling
            indices = np.random.randint(0, self.size, size=batch_size)
            
            # Generate random noise for consistency with standard buffer
            noises = torch.randn(batch_size, self.horizon_steps, self.action_dim, device=self.device)
            
            # Default n_step=1 and data source for online data
            n_steps_tensor = torch.ones(batch_size, 1, device=self.device, dtype=torch.long)
            data_source = torch.zeros(batch_size, 1, device=self.device, dtype=torch.long)
            
            if self.log_q_overestimation:
                return (
                    self.chunk_states[indices],      # (batch_size, cond_steps, obs_dim)
                    noises,                          # (batch_size, horizon_steps, action_dim)
                    self.chunk_actions[indices],     # (batch_size, horizon_steps, action_dim)
                    self.chunk_rewards[indices],     # (batch_size, 1)
                    self.chunk_next_states[indices], # (batch_size, cond_steps, obs_dim)
                    self.chunk_dones[indices],       # (batch_size, 1)
                    self.chunk_mc_returns[indices],  # (batch_size, 1)
                    n_steps_tensor,                  # (batch_size, 1)
                    data_source,                     # (batch_size, 1)
                )
            else:
                return (
                    self.chunk_states[indices],      # (batch_size, cond_steps, obs_dim)
                    noises,                          # (batch_size, horizon_steps, action_dim)
                    self.chunk_actions[indices],     # (batch_size, horizon_steps, action_dim)
                    self.chunk_rewards[indices],     # (batch_size, 1)
                    self.chunk_next_states[indices], # (batch_size, cond_steps, obs_dim)
                    self.chunk_dones[indices],       # (batch_size, 1)
                    n_steps_tensor,                  # (batch_size, 1)
                    data_source,                     # (batch_size, 1)
                )
    
    def _sample_rlpd(self, batch_size: int, expert_ratio: Optional[float] = None) -> Tuple[torch.Tensor, ...]:
        """RLPD symmetric sampling: mix expert data with online data."""
        current_expert_ratio = expert_ratio if expert_ratio is not None else self.expert_ratio
        
        # Calculate split
        expert_batch_size = int(batch_size * current_expert_ratio)
        online_batch_size = batch_size - expert_batch_size
        
        # Handle edge case where expert_batch_size becomes 0
        if expert_batch_size == 0:
            # No expert data needed, sample only online data
            if self.size > 0:
                return self._sample_standard(batch_size)
            else:
                raise ValueError("Cannot sample: no online data available and expert_batch_size=0")
        
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
    
    def get_batch_from_indices(self, indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get batch data given indices for flow matching training.
        Matches the format of pretraining dataset.
        
        Args:
            indices: Tensor of indices to sample
            
        Returns:
            Tuple of (states, actions) for flow matching training:
            - states: (batch_size, cond_steps, obs_dim) - conditions
            - actions: (batch_size, horizon_steps, action_dim) - trajectories
        """
        # Return in same format as pretraining dataset __getitem__
        return (
            self.chunk_states[indices],   # conditions: (batch_size, cond_steps, obs_dim)
            self.chunk_actions[indices],  # trajectories: (batch_size, horizon_steps, action_dim)
        )
    
    def _sample_expert_data(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample from expert dataset and convert to same format as replay buffer."""
        if batch_size == 0:
            raise ValueError("Cannot sample expert data with batch_size=0")
        
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
        """Return the total number of chunks (same as size for chunk storage)."""
        return self.size
    
    def is_full(self) -> bool:
        """Check if the buffer is full."""
        return self.size == self.max_size
    
    def clear(self):
        """Clear the replay buffer."""
        self.ptr = 0
        self.size = 0
        self.ongoing_episodes = [None for _ in range(self.n_envs)]
        log.info("HybridReplayBuffer cleared")