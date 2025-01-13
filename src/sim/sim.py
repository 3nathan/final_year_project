import mujoco
import time
import numpy as np

class Sim():
    def __init__(self, model_path):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

    # implement a parameterised camera
    def render_image(self, time=0):
        if time == None:
            time = 0

        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        with mujoco.Renderer(self.model) as renderer:
            while self.data.time < time:
                self.update_control()
                mujoco.mj_step(self.model, self.data)

            renderer.update_scene(self.data)

            return renderer.render()

    # implement a parameterised camera
    def render_video(self, duration=3, framerate=60):
        if duration == None:
            duration = 3

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
