from gym import spaces
from env.pusht.pusht_env import PushTEnv
import numpy as np
import cv2

class PushTImageEnv(PushTEnv):
    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}

    def __init__(self,
            legacy=False,
            block_cog=None, 
            damping=None,
            render_size=96,
            success_threshold=0.7695):
        super().__init__(
            legacy=legacy, 
            block_cog=block_cog,
            damping=damping,
            render_size=render_size,
            render_action=False,
            success_threshold=success_threshold)
        ws = self.window_size
        self.observation_space = spaces.Dict({
            'rgb': spaces.Box(
                low=0,
                high=255,
                shape=(render_size,render_size,3),
                dtype=np.uint8
            ),
            'state': spaces.Box(
                low=0,
                high=ws,
                shape=(2,),
                dtype=np.float32
            )
        })
        self.render_cache = None
    
    def _get_obs(self):
        img = super()._render_frame(mode='rgb_array')

        agent_pos = np.array(self.agent.position)
        # img_obs = np.moveaxis(img.astype(np.float32) / 255, -1, 0)
        obs = {
            'rgb': img,  # Use 'rgb' key to match wrapper expectations
            'state': agent_pos
        }
        # draw action
        if self.latest_action is not None:
            action = np.array(self.latest_action)
            coord = (action / 512 * 96).astype(np.int32)
            marker_size = int(8/96*self.render_size)
            thickness = int(1/96*self.render_size)
            cv2.drawMarker(img, coord,
                color=(255,0,0), markerType=cv2.MARKER_CROSS,
                markerSize=marker_size, thickness=thickness)
        self.render_cache = img
        return obs

    def _set_state(self, state):
        # Clear render cache when state changes
        self.render_cache = None
        return super()._set_state(state)
    
    def render(self, mode):
        assert mode == 'rgb_array'

        if self.render_cache is None:
            self._get_obs()
        
        return self.render_cache