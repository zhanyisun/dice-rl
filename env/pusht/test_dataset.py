from env.pusht.replay_buffer import ReplayBuffer
import numpy as np
import zarr

# zarr_path = 'data_dir/pusht/pusht/pusht_cchi_v7_replay.zarr'

# # path = 'data_dir/pusht/pusht/pusht_cchi_v7_replay.zarr'   # usually a directory ending with .zarr

# # root = zarr.open_group(path, mode="r")

# # # top-level keys (arrays + subgroups)
# # print(list(root.keys()))

# # # or, separately:
# # print("arrays:", list(root.array_keys()))
# # print("groups:", list(root.group_keys()))

# # # quick overview of the hierarchy
# # print(root.tree())

# # exit()
# replay_buffer = ReplayBuffer.copy_from_path(
#             zarr_path, keys=['img', 'state', 'action'])

# img = replay_buffer['img']
# state = replay_buffer['state']
# episode_end = replay_buffer.episode_ends[:]
# print('replay_buffer ', replay_buffer.keys())
# print('episode_end ', episode_end)
# print('img shape ', img.shape)
# print('img range ', np.max(img), np.min(img))
# print('state shape ', state.shape)

# square_data = np.load('data_dir/robomimic/square-img/ph/train.npz')
# rgb = square_data['images']
# traj_lengths = square_data['traj_lengths']


# print('square_data ', list(square_data.keys()))
# print('traj_lengths ', traj_lengths)
# print('rgb shape ', rgb.shape)
# print('rgb range ', np.max(rgb), np.min(rgb), rgb.dtype)

square_data = np.load('data_dir/pusht/processed/train.npz')
rgb = square_data['images']
traj_lengths = square_data['traj_lengths']


print('pusht_data ', list(square_data.keys()))
print('traj_lengths ', traj_lengths)
print('rgb shape ', rgb.shape)
print('rgb range ', np.max(rgb), np.min(rgb), rgb.dtype)
