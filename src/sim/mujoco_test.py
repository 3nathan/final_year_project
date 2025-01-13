#!/Users/nathan/imperial/fourth_year/fyp/venv/bin/python3
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

model = mujoco.MjModel.from_xml_path('../models/genghis.xml')
data = mujoco.MjData(model)

# enable joint visualization option:
# scene_option = mujoco.MjvOption()
# scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

duration = 10  # (seconds)
framerate = 60  # (Hz)

# Simulate and display video.
frames = []
mujoco.mj_resetData(model, data)

data.ctrl[6] = 45
# data.ctrl[7] = 45
# data.ctrl[8] = 45

with mujoco.Renderer(model) as renderer:
  while data.time < duration:
    mujoco.mj_step(model, data)

    if len(frames) < data.time * framerate:
      # renderer.update_scene(data, scene_option=scene_option)
      # renderer.update_scene(data, camera="zoom_plan")
      # renderer.update_scene(data, camera="plan")
      renderer.update_scene(data)
      pixels = renderer.render()
      frames.append(pixels)

# with mujoco.Renderer(model) as renderer:
#   mujoco.mj_forward(model, data)
#   # renderer.update_scene(data, camera="plan")
#   # renderer.update_scene(data, camera="zoom_plan")
#   # renderer.update_scene(data, camera="reg")
#   renderer.update_scene(data)

#   img = renderer.render()

# W = len(img[0])
# H = len(img)
# display = Display(W, H)

# display.show_img(img)

W = len(frames[0][0])
H = len(frames[0])
display = Display(W, H)

display.show_video(frames)
