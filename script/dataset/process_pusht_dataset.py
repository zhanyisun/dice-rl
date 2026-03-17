"""
Process PushT dataset from zarr format to our custom npz format for diffusion training.

The PushT dataset contains:
- state: (5,) [agent_x, agent_y, block_x, block_y, block_angle]
- action: (2,) [target_x, target_y]
- episode_ends: array of trajectory end indices

We render our own images instead of using the dataset's images.

Output format:
- states: concatenated state observations
- actions: concatenated actions
- rewards: computed rewards based on block coverage
- terminals: episode termination flags
- traj_lengths: length of each trajectory
- images: (N, 3, 96, 96) uint8 rendered images
"""

import numpy as np
import os
import logging
from tqdm import tqdm
import zarr
# import sys
import datetime

# Add parent directory to path for imports
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from env.pusht.pusht_env import PushTEnv
from env.pusht.pusht_image_env import PushTImageEnv
from env.pusht.replay_buffer import ReplayBuffer


def setup_logging(save_dir, save_name_prefix=""):
    """Setup logging to file and console, matching robomimic format."""
    log_path = os.path.join(
        save_dir,
        save_name_prefix + f"_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log",
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(),
        ],
    )
    return log_path


def process_pusht_dataset(zarr_path, save_dir, val_split=0.1, save_name_prefix="", success_threshold=0.95, normalize=True):
    """
    Process PushT dataset from zarr format to npz format.
    
    Args:
        zarr_path: Path to zarr dataset file
        save_dir: Directory to save processed dataset
        val_split: Fraction of data to use for validation
        save_name_prefix: Prefix for saved files
        success_threshold: Reward threshold for considering episode successful
        normalize: Whether to normalize states and actions to [-1, 1]
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Setup logging
    log_path = setup_logging(save_dir, save_name_prefix)
    logging.info(f"Log file: {log_path}")
    logging.info(f"Processing PushT dataset from: {zarr_path}")
    logging.info(f"Saving to: {save_dir}")
    logging.info(f"Validation split: {val_split}")
    logging.info(f"Success threshold: {success_threshold}")
    logging.info(f"Normalize: {normalize}")
    
    # Load zarr dataset
    logging.info(f"Loading dataset from {zarr_path}")
    replay_buffer = ReplayBuffer.copy_from_path(
        zarr_path, keys=['state', 'action']  # Don't load img since we'll render our own
    )
    
    # Access the zarr data directly
    root = zarr.open(zarr_path, 'r')
    # TODO: this is for test 
    # episode_ends = root['meta']['episode_ends'][:10]
    episode_ends = root['meta']['episode_ends'][:]
    states_data = root['data']['state'][:]
    actions_data = root['data']['action'][:]
    
    logging.info(f"Dataset shape - states: {states_data.shape}, actions: {actions_data.shape}")
    logging.info(f"Number of episodes: {len(episode_ends)}")
    
    # Initialize PushT environment for reward computation and rendering
    env = PushTImageEnv(render_size=96)
    
    # Process each trajectory
    all_states = []
    all_actions = []
    all_rewards = []
    all_terminals = []
    all_images = []
    traj_lengths = []
    
    truncated_episodes = 0
    total_truncated_steps = 0
    
    prev_end = 0
    for ep_idx, ep_end in enumerate(tqdm(episode_ends, desc="Processing trajectories")):
        # Extract trajectory data
        traj_states = states_data[prev_end:ep_end]
        traj_actions = actions_data[prev_end:ep_end]
        
        original_length = len(traj_states)
        
        # Compute rewards and render images for each timestep
        traj_rewards = []
        traj_terminals = []
        traj_images = []
        truncate_at = None
        
        for t in range(len(traj_states)):
            # Set environment state
            env._setup()  # Reset environment internals
            env._set_state(traj_states[t])
            
            # Compute reward
            reward, done = env._compute_reward()
            traj_rewards.append(reward)
            
            # Render image
            img = env.render(mode='rgb_array')  # Returns (96, 96, 3) uint8
            # Convert from HWC to CHW format
            img_chw = np.transpose(img, (2, 0, 1))  # (3, 96, 96)
            traj_images.append(img_chw)
            
            # Check for early termination using our threshold
            if reward > success_threshold and truncate_at is None:
                truncate_at = t + 1  # Include this successful state
                traj_terminals.append(True)
            else:
                traj_terminals.append(False)
        
        # Truncate trajectory if success was achieved
        if truncate_at is not None:
            traj_states = traj_states[:truncate_at]
            traj_actions = traj_actions[:truncate_at]
            traj_images = traj_images[:truncate_at]
            traj_rewards = traj_rewards[:truncate_at]
            traj_terminals = traj_terminals[:truncate_at]
            
            truncated_episodes += 1
            truncated_steps = original_length - truncate_at
            total_truncated_steps += truncated_steps
            
            if ep_idx < 300:  # Log first few episodes
                logging.info(f"Episode {ep_idx}: truncated at step {truncate_at}/{original_length} (removed {truncated_steps} steps, reward={traj_rewards[truncate_at-1]:.3f})")
        else:
            # No success achieved
            max_reward = max(traj_rewards)
            if ep_idx < 300:
                logging.info(f"Episode {ep_idx}: no truncation, max reward = {max_reward:.3f}")
        
        traj_images = np.array(traj_images, dtype=np.uint8)  # (T, 3, 96, 96)
        
        # Verify that only the last step has terminals=True (if any)
        traj_terminals_array = np.array(traj_terminals)
        # if traj_terminals_array.any():  # If there's any True value
        #     true_indices = np.where(traj_terminals_array)[0]
        #     if len(true_indices) != 1 or true_indices[0] != len(traj_terminals) - 1:
        #         logging.warning(f"Episode {ep_idx}: terminals=True not only at last step! True at indices: {true_indices}")
        #         # Fix it to only have terminal at the last step
        #         traj_terminals_array[:] = False
        #         traj_terminals_array[-1] = True
        
        # Store trajectory data
        all_states.append(traj_states)
        all_actions.append(traj_actions)
        all_rewards.append(np.array(traj_rewards))
        all_terminals.append(traj_terminals_array)
        all_images.append(traj_images)
        traj_lengths.append(len(traj_states))
        
        prev_end = ep_end
    
    # Calculate total transitions before truncation
    original_total_steps = prev_end  # Total steps in original dataset
    truncated_total_steps = sum(traj_lengths)  # Total steps after truncation
    
    # # Final validation: check that terminals are only True at trajectory ends
    # for ep_idx, (terminals, traj_len) in enumerate(zip(all_terminals, traj_lengths)):
    #     if len(terminals) != traj_len:
    #         logging.error(f"Episode {ep_idx}: length mismatch! terminals={len(terminals)}, traj_len={traj_len}")
    #     if terminals.any():
    #         true_indices = np.where(terminals)[0]
    #         if len(true_indices) != 1 or true_indices[0] != len(terminals) - 1:
    #             logging.error(f"Episode {ep_idx}: Invalid terminals! True at indices: {true_indices}, should be only at {len(terminals)-1}")
    
    # Calculate min/max for all data (before split)
    all_states_concat = np.concatenate(all_states, axis=0)
    all_actions_concat = np.concatenate(all_actions, axis=0)
    obs_min = all_states_concat.min(axis=0)
    obs_max = all_states_concat.max(axis=0)
    action_min = all_actions_concat.min(axis=0)
    action_max = all_actions_concat.max(axis=0)
    
    # Report basic statistics (matching robomimic format)
    logging.info("===== Basic stats =====")
    logging.info(f"Total transitions: {truncated_total_steps}")
    logging.info(f"Total trajectories: {len(traj_lengths)}")
    logging.info(
        f"Traj length mean/std: {np.mean(traj_lengths)}, {np.std(traj_lengths)}"
    )
    logging.info(
        f"Traj length min/max: {np.min(traj_lengths)}, {np.max(traj_lengths)}"
    )
    logging.info(f"obs min: {obs_min}")
    logging.info(f"obs max: {obs_max}")
    logging.info(f"action min: {action_min}")
    logging.info(f"action max: {action_max}")
    
    # Normalize states and actions to [-1, 1] if requested
    if normalize:
        logging.info("Normalizing states and actions to [-1, 1]")
        for i in range(len(all_states)):
            # Normalize states: 2 * (x - min) / (max - min + eps) - 1
            all_states[i] = 2 * (all_states[i] - obs_min) / (obs_max - obs_min + 1e-6) - 1
            # Normalize actions
            all_actions[i] = 2 * (all_actions[i] - action_min) / (action_max - action_min + 1e-6) - 1
    
    # Split into train/val
    num_episodes = len(traj_lengths)
    num_val = int(num_episodes * val_split)
    num_train = num_episodes - num_val
    
    # Use deterministic split (last episodes for validation)
    train_indices = list(range(num_train))
    val_indices = list(range(num_train, num_episodes))
    
    # Prepare train and val datasets
    def prepare_dataset(indices):
        dataset_states = []
        dataset_actions = []
        dataset_rewards = []
        dataset_terminals = []
        dataset_images = []
        dataset_traj_lengths = []
        
        for idx in indices:
            dataset_states.append(all_states[idx])
            dataset_actions.append(all_actions[idx])
            dataset_rewards.append(all_rewards[idx])
            dataset_terminals.append(all_terminals[idx])
            dataset_images.append(all_images[idx])
            dataset_traj_lengths.append(traj_lengths[idx])
        
        # Concatenate all trajectories
        return {
            "states": np.concatenate(dataset_states, axis=0),
            "actions": np.concatenate(dataset_actions, axis=0),
            "rewards": np.concatenate(dataset_rewards, axis=0),
            "terminals": np.concatenate(dataset_terminals, axis=0),
            "images": np.concatenate(dataset_images, axis=0),
            "traj_lengths": np.array(dataset_traj_lengths)
        }
    
    train_data = prepare_dataset(train_indices)
    val_data = prepare_dataset(val_indices) if num_val > 0 else None
    
    # Log truncation statistics first (matching robomimic format)
    logging.info("===== Truncation Statistics =====")
    logging.info(f"Original total steps: {original_total_steps}")
    logging.info(f"Truncated total steps: {truncated_total_steps}")
    logging.info(f"Reduction: {original_total_steps - truncated_total_steps} steps ({100 * (1 - truncated_total_steps/original_total_steps):.1f}%)")
    
    # Log train/val split info (matching robomimic format)
    logging.info(
        f"Train - Trajectories: {len(train_data['traj_lengths'])}, Transitions: {np.sum(train_data['traj_lengths'])}"
    )
    if val_data is not None:
        logging.info(
            f"Val - Trajectories: {len(val_data['traj_lengths'])}, Transitions: {np.sum(val_data['traj_lengths'])}"
        )
    
    # Log image statistics (matching robomimic format)
    train_images_shape = train_data["images"].shape
    logging.info(f"Train images shape: {train_images_shape}")
    logging.info(f"Image value range: [{train_data['images'].min()}, {train_data['images'].max()}]")
    # PushT has single camera, so we log that
    logging.info(f"Image channels match expected: {train_images_shape[1]} = 1 cameras × 3 RGB")
    
    # Save datasets
    train_save_path = os.path.join(save_dir, save_name_prefix + "train.npz")
    np.savez_compressed(train_save_path, **train_data)
    
    if val_data is not None:
        val_save_path = os.path.join(save_dir, save_name_prefix + "val.npz")
        np.savez_compressed(val_save_path, **val_data)
    
    # Save normalization statistics in robomimic format
    # Note: we already calculated obs_min/max and action_min/max above from all data
    normalization_save_path = os.path.join(
        save_dir, save_name_prefix + "normalization.npz"
    )
    np.savez_compressed(
        normalization_save_path,
        obs_min=obs_min,
        obs_max=obs_max,
        action_min=action_min,
        action_max=action_max
    )
    
    # Close environment
    env.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr_path", type=str, 
                       default="data_dir/pusht/pusht/pusht_cchi_v7_replay.zarr",
                       help="Path to zarr dataset")
    parser.add_argument("--save_dir", type=str,
                       default="data_dir/pusht/processed_truncated_1",
                       help="Directory to save processed dataset")
    parser.add_argument("--val_split", type=float, default=0.0,
                       help="Fraction of data for validation")
    parser.add_argument("--save_name_prefix", type=str, default="",
                       help="Prefix for saved files")
    parser.add_argument("--success_threshold", type=float, default=0.7695,
                       help="Reward threshold for considering episode successful")
    parser.add_argument("--normalize", action="store_true", default=True,
                       help="Normalize states and actions to [-1, 1]")
    
    args = parser.parse_args()
    
    process_pusht_dataset(args.zarr_path, args.save_dir, args.val_split, 
                         args.save_name_prefix, args.success_threshold, args.normalize)