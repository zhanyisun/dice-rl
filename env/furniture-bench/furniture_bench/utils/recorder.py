from typing import Dict
import imageio
import numpy as np
import torch


class VideoRecorder:
    def __init__(self, output_path, fps, width, height, channel_first=True):
        self.output_path = str(output_path)
        print(f"Recording video to {output_path}")
        self.fps = fps
        self.width = width
        self.height = height
        self.channel_first = channel_first
        self.writer = None
        self.record = False

    def start_recording(self):
        self.writer = imageio.get_writer(self.output_path, fps=self.fps)
        self.record = True

    def stop_recording(self):
        if self.writer is not None:
            self.writer.close()
        self.record = False

    def restart_recording(self):
        self.stop_recording()
        self.start_recording()

    def record_frame(self, obs: Dict[str, torch.Tensor]):
        if self.record:
            record_images = []
            for k in ["color_image1", "color_image2"]:
                img: torch.Tensor = obs[k][0].cpu().numpy()
                if self.channel_first:
                    img = img.transpose(0, 2, 3, 1)
                record_images.append(img.squeeze())
            stacked_img = np.hstack(record_images)
            self.writer.append_data(stacked_img)
