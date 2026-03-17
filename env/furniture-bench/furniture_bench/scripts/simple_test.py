# from furniture_bench.envs.furniture_sim_env import FurnitureSimEnv
# try:
#     env = FurnitureSimEnv(
#         furniture='one_leg',
#         gpu_id=0,
#         headless=True,
#         num_envs=1
#     )
#     print('FurnitureSimEnv created successfully')
# except Exception as e:
#     print(f'FurnitureSimEnv error: {e}')
#     import traceback
#     traceback.print_exc()

from furniture_bench.envs.furniture_rl_sim_env import FurnitureRLSimEnv
try:
    env = FurnitureRLSimEnv(
        furniture='one_leg',
        act_rot_repr='rot_6d',
        action_type='pos',
        april_tags=False,
        concat_robot_state=True,
        ctrl_mode='diffik',
        gpu_id=0,
        headless=True,
        num_envs=1,
        observation_space='state',
        randomness='low',
        max_env_steps=700,
        record=False,
        pos_scalar=1,
        rot_scalar=1,
        stiffness=1000,
        damping=200,
    )
    print('DPPO-style FurnitureRLSimEnv created successfully!')
    obs = env.reset()
    print('Reset successful!')
    env.close()
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()