import numpy as np
from PIL import Image
import os
os.environ['MUJOCO_GL'] = 'egl'

import gymnasium
import ogbench
import mujoco
import h5py
import numpy as np

env = gymnasium.make(
    "visual-cube-single-v0",
    terminate_at_goal=False,
    mode="data_collection",
)
env.reset()
img = env.unwrapped.render(camera="front_pixels")
Image.fromarray(img).save(
    f"first_source_state.png"
)

print("got mujoco")
