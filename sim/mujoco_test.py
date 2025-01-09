#!/Users/nathan/imperial/fourth_year/fyp/split_brain/bin/python3
import mujoco
import os
import subprocess

# other imports and helper functions
import time
import itertools
import numpy as np

# graphics and plotting
from display import Display
import matplotlib.pyplot as plt

# img = np.random.rand(400,400, 3)*255
# display = Display()
# display.show_img(img)
# time.sleep(1)
# img = np.random.rand(400,400, 3)*255
# display.show_img(img)
# time.sleep(1)
# display.handle_close()

xml = """
<mujoco>
  <worldbody>
    <light name="top" pos="0 0 1"/>
    <body name="box_and_sphere" euler="0 0 -30">
      <joint name="swing" type="hinge" axis="1 -1 0" pos="-.2 -.2 -.2"/>
      <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
      <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

# enable joint visualization option:
scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

duration = 3.8  # (seconds)
framerate = 60  # (Hz)

# Simulate and display video.
frames = []
mujoco.mj_resetData(model, data)
with mujoco.Renderer(model) as renderer:
  while data.time < duration:
    mujoco.mj_step(model, data)
    if len(frames) < data.time * framerate:
      renderer.update_scene(data, scene_option=scene_option)
      pixels = renderer.render()
      frames.append(pixels)

W = len(frames[0])
H = len(frames[0][0])
display = Display(W, H)

display.show_video(frames, fps=framerate)
