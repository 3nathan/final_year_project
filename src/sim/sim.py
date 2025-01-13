import mujoco
import time
import numpy as np

class Sim():
    def __init__(self, model_path):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
    
    def render_video(self, duration=10, framerate=60, camera=None):
        frames = []
        mujoco.mj_resetData(self.model, self.data)

        with mujoco.Renderer(self.model) as renderer:
            while self.data.time < duration:
                self.update_control()

                mujoco.mj_step(self.model, self.data)

                if len(frames) < self.data.time * framerate:
                    if camera:
                        renderer.update_scene(self.data, camera=camera)
                    else:
                        renderer.update_scene(self.data)
                    pixels = renderer.render()
                    frames.append(pixels)

        return frames

    def update_control(self):
        self.data.ctrl[6] = 45
        self.data.ctrl[7] = 45
        self.data.ctrl[8] = 45
        self.data.ctrl[9] = 45
        self.data.ctrl[10] = 45
        self.data.ctrl[11] = 45
