import numpy as np
import torch
import mujoco
from mujoco import MjModel, MjData

from util.config import Config

# needed for the reward functionm stuff
import math

CONFIG = Config()

class SimEnv():
    def __init__(self, model_path, seed=None):
        print(f"Loading MuJoCo model: {model_path} into the simulation environment")
        self.model = MjModel.from_xml_path(model_path)
        self.data = MjData(self.model)

        # adjust following based on specific actuators and state variables
        self.action_dim = self.model.nu
        self.obs_dim = self.model.nq + self.model.nv

        self.rng = np.random.default_rng(seed)

        print("Simulation environment initialised")

        # hard coded for genghis reward function
        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "genghis")
        self.latent_velocity = np.zeros(2)
        # hard coded for genghis reward function

    def step(self, action):
        """
        Apply the action, step the simulation forward and return observation, reward, done, info
        """
        # ensure action is within control limits
        action_np = action.detach().cpu().numpy()
        action_clipped = np.clip(
            action_np,
            self.model.actuator_ctrlrange[:, 0],
            self.model.actuator_ctrlrange[:, 1]
        )
        self.data.ctrl[:] = action_clipped

        mujoco.mj_step(self.model, self.data)
        observation = self._get_obs()
        reward = self._compute_reward(observation, action)
        done = self._check_done(observation)
        
        # info dictionary can include diagnostic data
        info = {}

        return observation, reward, done, info

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        # hard coded for genghis reward function
        self.R = self.data.xmat[self.body_id].reshape(3, 3)
        self.initial_height = 0.1
        self.initial_yaw = math.atan2(self.R[1, 0], self.R[0, 0])
        # hard coded for genghis reward function
        self._apply_reset_noise()
        return self._get_obs()

    def set_seed(self):
        """
        Set the random seed for reproducibility
        """
        self.rng = np.random.default_rng(seed)

    def _get_obs(self):
        """
        Construct the observation from the MuJoCo data
        """
        return np.concatenate([self.data.qpos.ravel(), self.data.qvel.ravel()])

    # customise to reflect the objectives of the task
    def _compute_reward(self, observation, action):
        """
        Compute reward based on the current observation and action
        """
        # reward should be in terms of compliance with the gait
        # gait parameters can be randomised
        # gait is defined as the following vectors concatenated:
        #   x_body
        #   x_dot_body
        #   theta_dot_body
        # the reward function is hard coded for genghis here:
        R = self.data.xmat[self.body_id].reshape(3, 3)
        yaw = math.atan2(R[1, 0], R[0, 0])
        height = self.data.xpos[self.body_id][2]
        v_global = self.data.cvel[self.body_id][:3]
        v_local = R.T @ v_global
        v_2d_local = v_local[:2]  # [forward/backward, left/right]

        vel_error = np.linalg.norm(v_2d_local - self.latent_velocity) # define latent velocity
        reward_vel = np.exp(-vel_error**2)

        yaw_error = np.abs(yaw - self.initial_yaw)
        yaw_error = (yaw_error + np.pi) % (2 * np.pi) - np.pi
        reward_yaw = np.exp(-yaw_error**2/0.01)

        height_error = np.abs(height - self.initial_height)
        reward_height = np.exp(-height_error**2 / 0.01)

        reward = (
            0.6 * reward_vel +
            0.2 * reward_yaw +
            0.2 * reward_height
        )
        
        return reward

    # modify to define when an episode should end
    def _check_done(self, observation):
        """
        Check if the episode should terminate
        """
        return self.data.time >= CONFIG.EPISODE_DURATION

    def _apply_reset_noise(self):
        """
        Add random noise to initial positions and velocities
        """
        noise_scale = 0.01
        self.data.qpos[:] += self.rng.normal(scale=noise_scale, size=self.data.qpos.shape)
        self.data.qvel[:] += self.rng.normal(scale=noise_scale, size=self.data.qvel.shape)

    def get_dims(self):
        return self.obs_dim, self.action_dim

    # implement a parameterised camera
    # def render_image(self, time=0):
    #     # VVV ad hoc fix VVV
    #     if time == None:
    #         time = 0
    #     # ^^^ ad hoc fix ^^^

    #     mujoco.mj_resetData(self.model, self.data)
    #     mujoco.mj_forward(self.model, self.data)

    #     with mujoco.Renderer(self.model) as renderer:
    #         while self.data.time < time:
    #             self.update_control()
    #             mujoco.mj_step(self.model, self.data)

    #         renderer.update_scene(self.data)

    #         return renderer.render()

    # # implement a parameterised camera
    # def render_video(self, duration=3, framerate=60, camera=None):
    #     # VVV ad hoc fix VVV
    #     if duration == None:
    #         duration = 3
    #     # ^^^ ad hoc fix ^^^

    #     frames = []
    #     mujoco.mj_resetData(self.model, self.data)

    #     with mujoco.Renderer(self.model) as renderer:
    #         while self.data.time < duration:
    #             self.update_control()
    #             mujoco.mj_step(self.model, self.data)

    #             if len(frames) < self.data.time * framerate:
    #                 if camera:
    #                     renderer.update_scene(self.data, camera=camera)
    #                 else:
    #                     renderer.update_scene(self.data)
    #                 pixels = renderer.render()
    #                 frames.append(pixels)

    #     return frames

    # def load_control(self):
    #     if self.model_name == "genghis":
    #     #     self.control = Genghis()
    #         self.state_variables = np.zeros(1 + 3*3 + (self.model.njnt-1)*2 + 6)

    #         for i in range(self.model.ngeom):
    #             # print(self.model.geom_names)
    #             # if "ground" in self.model.names[i]:
    #             #     self.ground_id = i

    #             #     break

    #     return 0

    # def get_state_variables(self):
    #     # state variables: [body height, body theta, body dx/dt, body dtheta/dt, joint theta, joint dtheta/dt, joint contacts with floor]
    #     body_id = self.model.body("genghis").id
    #     self.state_variables[0] = self.data.xpos[body_id][2]
    #     self.state_variables[1:5] = self.data.xquat[body_id]
    #     self.state_variables[5:11] = self.data.cvel[body_id]

    #     for i in range(1, self.model.njnt):
    #         joint_pos = self.data.joint(i).qpos
    #         joint_vel = self.data.joint(i).qvel
    #         # print(f"Joint {i} ({joint_name}) pos: {joint_pos}")
    #         # print(i)
    #         self.state_variables[2*3 + 4 + i] = joint_pos
    #         self.state_variables[2*3 + 4 - 1 + i + self.model.njnt] = joint_vel

    #     # for i in range(self.model.ngeom):
    #     #     if "leg" in self.model.names[i]:
                

    #     # print(self.model.njnt)

    #     # print(self.state_variables[1:4])
    #     # print(self.state_variables)

    # def update_control(self):
    #     self.get_state_variables()

    #     self.data.ctrl[6] = 45
    #     self.data.ctrl[7] = 45
    #     self.data.ctrl[8] = 45
    #     self.data.ctrl[9] = 45
    #     self.data.ctrl[10] = 45
    #     self.data.ctrl[11] = 45
