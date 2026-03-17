import os
import json

try:
    from collections.abc import Iterable
except ImportError:
    Iterable = (tuple, list)


def make_async(
    id,
    num_envs=1,
    asynchronous=True,
    wrappers=None,
    render=False,
    obs_dim=23,
    action_dim=7,
    env_type=None,
    max_episode_steps=None,
    # below for furniture only
    gpu_id=0,
    headless=True,
    record=False,
    normalization_path=None,
    furniture="one_leg",
    randomness="low",
    obs_steps=1,
    act_steps=8,
    sparse_reward=False,
    # below for robomimic only
    robomimic_env_cfg_path=None,
    use_image_obs=False,
    render_offscreen=False,
    reward_shaping=False,
    shape_meta=None,
    # below for pusht only
    success_threshold=0.7695,
    **kwargs,
):
    """Create a vectorized environment from multiple copies of an environment,
    from its id.

    Parameters
    ----------
    id : str
        The environment ID. This must be a valid ID from the registry.

    num_envs : int
        Number of copies of the environment.

    asynchronous : bool
        If `True`, wraps the environments in an :class:`AsyncVectorEnv` (which uses
        `multiprocessing`_ to run the environments in parallel). If ``False``,
        wraps the environments in a :class:`SyncVectorEnv`.

    wrappers : dictionary, optional
        Each key is a wrapper class, and each value is a dictionary of arguments

    Returns
    -------
    :class:`gym.vector.VectorEnv`
        The vectorized environment.

    Example
    -------
    >>> env = gym.vector.make('CartPole-v1', num_envs=3)
    >>> env.reset()
    array([[-0.04456399,  0.04653909,  0.01326909, -0.02099827],
           [ 0.03073904,  0.00145001, -0.03088818, -0.03131252],
           [ 0.03468829,  0.01500225,  0.01230312,  0.01825218]],
          dtype=float32)
    """

    if env_type == "pusht":
        # Handle PushT environment creation
        from env.gym_utils.async_vector_env import AsyncVectorEnv
        from env.gym_utils.sync_vector_env import SyncVectorEnv
        
        def _make_pusht_env():
            # Let wrappers handle environment creation (like robomimic does)
            # Start with None env - the pusht_state or pusht_image wrapper will create it
            env = None
            
            # Apply ALL wrappers from config
            if wrappers is not None:
                from env.gym_utils.wrapper import wrapper_dict
                for wrapper, args in wrappers.items():
                    env = wrapper_dict[wrapper](env, **args)
            
            return env
        
        # Create dummy environment function for PushT
        def dummy_pusht_env_fn():
            import gym
            from gym import spaces
            import numpy as np
            from env.gym_utils.wrapper.multi_step import MultiStep
            
            # Create a fake env for metadata
            env = gym.Env()
            observation_space = spaces.Dict()
            
            if use_image_obs:
                # Image observations: both state and rgb
                # For image-based, state is only 2D (agent position)
                observation_space["state"] = spaces.Box(
                    low=-1, high=1, shape=(2,), dtype=np.float32
                )
                observation_space["rgb"] = spaces.Box(
                    low=0, high=255, shape=(96, 96, 3), dtype=np.uint8
                )
            else:
                # State-only observations
                observation_space["state"] = spaces.Box(
                    low=-1, high=1, shape=(5,), dtype=np.float32
                )
            
            env.observation_space = observation_space
            env.action_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
            env.metadata = {
                "render.modes": ["human", "rgb_array"],
                "video.frames_per_second": 30,
            }
            
            # Apply multi_step wrapper if configured
            if wrappers is not None and 'multi_step' in wrappers:
                return MultiStep(env=env, n_obs_steps=wrappers['multi_step']['n_obs_steps'])
            return env
        
        # Create vectorized environments
        env_fns = [_make_pusht_env for _ in range(num_envs)]
        
        # Return async or sync vector env with proper dummy env
        return (
            AsyncVectorEnv(env_fns, dummy_env_fn=dummy_pusht_env_fn if use_image_obs else None)
            if asynchronous
            else SyncVectorEnv(env_fns)
        )
    
    if env_type == "furniture":
        from furniture_bench.envs.observation import DEFAULT_STATE_OBS
        from furniture_bench.envs.furniture_rl_sim_env import FurnitureRLSimEnv
        from env.gym_utils.wrapper.furniture import FurnitureRLSimEnvMultiStepWrapper

        env = FurnitureRLSimEnv(
            act_rot_repr="rot_6d",
            action_type="pos",
            april_tags=False,
            concat_robot_state=True,
            ctrl_mode="diffik",
            obs_keys=DEFAULT_STATE_OBS,
            furniture=furniture,
            gpu_id=gpu_id,
            headless=headless,
            num_envs=num_envs,
            observation_space="state",
            randomness=randomness,
            max_env_steps=max_episode_steps,
            record=record,
            pos_scalar=1,
            rot_scalar=1,
            stiffness=1_000,
            damping=200,
        )
        env = FurnitureRLSimEnvMultiStepWrapper(
            env,
            n_obs_steps=obs_steps,
            n_action_steps=act_steps,
            prev_action=False,
            reset_within_step=False,
            pass_full_observations=False,
            normalization_path=normalization_path,
            sparse_reward=sparse_reward,
        )
        return env

    # avoid import error due incompatible gym versions
    from gym import spaces
    from env.gym_utils.async_vector_env import AsyncVectorEnv
    from env.gym_utils.sync_vector_env import SyncVectorEnv
    from env.gym_utils.wrapper import wrapper_dict

    __all__ = [
        "AsyncVectorEnv",
        "SyncVectorEnv",
        "VectorEnv",
        "VectorEnvWrapper",
        "make",
    ]

    # import the envs
    if robomimic_env_cfg_path is not None:
        import robomimic.utils.env_utils as EnvUtils
        import robomimic.utils.obs_utils as ObsUtils
    elif "avoiding" in id:
        import gym_avoiding
    else:
        import d4rl.gym_mujoco
    from gym.envs import make as make_

    def _make_env():
        if robomimic_env_cfg_path is not None:
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
            os.environ["MUJOCO_GL"] = "egl"
            if render_offscreen or use_image_obs:
                # Ensure EGL uses the correct GPU device
                if "CUDA_VISIBLE_DEVICES" in os.environ:
                    # Map CUDA device to EGL device
                    cuda_device = os.environ["CUDA_VISIBLE_DEVICES"].split(',')[0]
                    os.environ["EGL_DEVICE_ID"] = cuda_device
                    os.environ["MUJOCO_EGL_DEVICE_ID"] = cuda_device
            with open(robomimic_env_cfg_path, "r") as f:
                env_meta = json.load(f)
            env_meta["reward_shaping"] = reward_shaping
            
            # Check if we should use absolute actions (from kwargs)
            if kwargs.get("abs_action", False):
                env_meta["env_kwargs"]["controller_configs"]["control_delta"] = False
            
            env = EnvUtils.create_env_from_metadata(
                env_meta=env_meta,
                render=render,
                # only way to not show collision geometry is to enable render_offscreen, which uses a lot of RAM.
                render_offscreen=render_offscreen,
                use_image_obs=use_image_obs,
                # render_gpu_device_id=0,
            )
            # Robosuite's hard reset causes excessive memory consumption.
            # Disabled to run more envs.
            # https://github.com/ARISE-Initiative/robosuite/blob/92abf5595eddb3a845cd1093703e5a3ccd01e77e/robosuite/environments/base.py#L247-L248
            env.env.hard_reset = False
        else:  # d3il, gym
            env = make_(id, render=render, **kwargs)

        # add wrappers
        if wrappers is not None:
            for wrapper, args in wrappers.items():
                env = wrapper_dict[wrapper](env, **args)
        return env

    def dummy_env_fn():
        """TODO(allenzren): does this dummy env allow camera obs for other envs besides robomimic?"""
        import gym
        import numpy as np
        from env.gym_utils.wrapper.multi_step import MultiStep
        from env.gym_utils.wrapper.multi_step_full import MultiStepFull

        # Avoid importing or using env in the main process
        # to prevent OpenGL context issue with fork.
        # Create a fake env whose sole purpose is to provide
        # obs/action spaces and metadata.
        env = gym.Env()
        observation_space = spaces.Dict()
        if shape_meta is not None:  # rn only for images
            for key, value in shape_meta["obs"].items():
                shape = value["shape"]
                if key.endswith("rgb"):
                    min_value, max_value = -1, 1
                elif key.endswith("state"):
                    min_value, max_value = -1, 1
                else:
                    raise RuntimeError(f"Unsupported type {key}")
                observation_space[key] = spaces.Box(
                    low=min_value,
                    high=max_value,
                    shape=shape,
                    dtype=np.float32,
                )
        else:
            observation_space["state"] = gym.spaces.Box(
                -1,
                1,
                shape=(obs_dim,),
                dtype=np.float32,
            )
        env.observation_space = observation_space
        env.action_space = gym.spaces.Box(-1, 1, shape=(action_dim,), dtype=np.int64)
        env.metadata = {
            "render.modes": ["human", "rgb_array", "depth_array"],
            "video.frames_per_second": 12,
        }
        # Handle both multi_step and multi_step_full (for upsampling)
        if 'multi_step' in wrappers:
            return MultiStep(env=env, n_obs_steps=wrappers.multi_step.n_obs_steps)
        elif 'multi_step_full' in wrappers:
            # Use MultiStep for dummy env even when MultiStepFull is configured
            # This is fine since dummy env is only for metadata
            return MultiStepFull(env=env, n_obs_steps=wrappers.multi_step_full.n_obs_steps)
        else:
            return env

    env_fns = [_make_env for _ in range(num_envs)]
    return (
        AsyncVectorEnv(
            env_fns,
            dummy_env_fn=(
                dummy_env_fn if render or render_offscreen or use_image_obs else None
            ),
            delay_init="avoiding" in id,  # add delay for D3IL initialization
        )
        if asynchronous
        else SyncVectorEnv(env_fns)
    )
