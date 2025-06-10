import numpy as np
import torch
from torch.distributions import Normal
import time
import mujoco
from mujoco import MjModel, MjData

from sim.display import Display
from util.config import Config

# needed for the reward functionm stuff
import math

CONFIG = Config()

class SimEnv():
    def __init__(self, model_path, seed=None, video=False):
        print(f"Loading MuJoCo model: {model_path} into the simulation environment")
        self.model = MjModel.from_xml_path(model_path)
        self.data = MjData(self.model)

        # hard coded for genghis reward function
        self.ground_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'plain')
        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'quadruped')
        if self.body_id == -1:
            self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'genghis')
        self.latent_control = np.zeros(CONFIG.GENGHIS_CTRL_DIM)
        # hard coded for genghis reward function

        self.render_video = video

        # adjust following based on specific actuators and state variables
        self.action_dim = self.model.nu
        self.obs_dim = len(self._get_obs())
        self.sensor_dim = self.model.nsensor

        self.rng = np.random.default_rng(seed)
        
        self._prev_sensor_data = np.zeros(self.model.nsensor)
        self._robot_steps = 0

        self._original_orientation=self.data.xquat[self.body_id]


        print("Simulation environment initialised")

    def step(self, action):
        """
        Apply the action, step the simulation forward and return observation, reward, done, info
        """
        # ensure action is within control limits
        action = action.detach().cpu().numpy()
        action_clipped = np.clip(action, 0, 1)
        action_scaled = action_clipped * (self.model.actuator_ctrlrange[:, 1] - self.model.actuator_ctrlrange[:, 0]) + self.model.actuator_ctrlrange[:, 0]
        self.data.ctrl[:] = action_scaled

        mujoco.mj_step(self.model, self.data)
        observation = self._get_obs()
        reward = self._compute_reward(observation, action)
        done = self._check_done(observation)

        info = {}

        return observation, reward, done, info

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
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
        filtered_sensor_data = np.zeros(self.model.nsensor)
        self._robot_steps = 0

        for i in range(self.model.nsensor):
            if self.data.sensordata[i] > 0:
                filtered_sensor_data[i] = 1
                if self._prev_sensor_data[i] == 0:
                    self._robot_steps +=1
        
        return np.concatenate([
            self.data.qpos.ravel(),
            self.data.qvel.ravel(),
            self.data.xquat[self.body_id].ravel(),
            self.data.xpos[self.body_id].ravel(),
            self.data.cvel[self.body_id].ravel(),
            filtered_sensor_data
        ])

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
        # R = self.data.xmat[self.body_id].reshape(3, 3)
        # yaw = math.atan2(R[1, 0], R[0, 0])
        # height = self.data.xpos[self.body_id][2]
        # v_global = self.data.cvel[self.body_id][:3]
        # v_global = self.data.cvel[3][:3]
        # v_local = R.T @ v_global
        # v_2d_local = v_local[:2]  # [forward/backward, left/right]

        # vel_error = np.linalg.norm(v_2d_local - self.latent_velocity) # define latent velocity
        # reward_vel = np.exp(-vel_error*vel_error)

        # yaw_error = np.abs(yaw - self.initial_yaw)
        # yaw_error = (yaw_error + np.pi) % (2 * np.pi) - np.pi
        # reward_yaw = np.exp(-100*yaw_error*yaw_error)

        # height_error = np.abs(height - self.initial_height)
        # reward_height = np.exp(-100*height_error*height_error)

        # reward = (
        #     0.6 * reward_vel +
        #     0.2 * reward_yaw +
        #     0.2 * reward_height
        # )

        # height = self.data.xpos[self.body_id][2]
        # height_error = height - 0.15
        # height_reward = np.exp(-100*height_error*height_error)

        velocity = self.data.cvel[self.body_id][3:5]

        speed = np.linalg.norm(velocity)
        speed_error = speed - np.linalg.norm(self.latent_control[0:2])
        speed_reward = np.exp(-100*speed_error*speed_error)

        direction = velocity / speed
        direction_alignment = np.dot(direction, self.data.xmat[self.body_id][0:2])-1
        direction_reward = np.exp(-100*direction_alignment*direction_alignment)

        # velocity_error = np.linalg.norm(velocity - [0.1, 0])
        # velocity_error = np.linalg.norm(velocity - self.latent_control[0:2])
        # velocity_reward = np.exp(-100*velocity_error*velocity_error)

        ang = self.data.cvel[self.body_id][2]
        ang_error = np.linalg.norm(ang - self.latent_control[2])
        ang_reward = np.exp(-100*ang_error*ang_error)

        # ang = self.data.cvel[self.body_id][0:3]
        # ang_error = np.linalg.norm(ang - [0, 0, 0])
        # ang_reward = np.exp(-100*ang_error*ang_error)

        # pos_error = np.linalg.norm(self.data.xpos[self.body_id][0:2] - [0, 0])
        # pos_reward = np.exp(-0.05*pos_error*pos_error)

        # orientation = self.data.xquat[self.body_id]
        # orientation_error = np.linalg.norm(orientation - self._original_orientation)
        # orientation_reward = np.exp(-100*orientation_error*orientation_error)

        # print(velocity_reward)
        # print(ang_reward)
        reward = (
            speed_reward +
            direction_reward +
            ang_reward
        )
        
        return np.float32(reward)

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
        self.data.xquat[self.body_id][:] += self.rng.normal(scale=noise_scale, size=self.data.xquat[self.body_id].shape)
        self.data.xpos[self.body_id] += self.rng.normal(scale=noise_scale, size=self.data.xpos[self.body_id].shape)
        self.data.cvel[self.body_id][:] += self.rng.normal(scale=noise_scale, size=self.data.cvel[self.body_id].shape)

    def get_dims(self):
        return self.obs_dim, self.action_dim

    # stand in demo loop
    def run_demo(self, policy=None, t=1/60):
        # Reset environment
        self._initialise_display()
        obs = self.reset()

        if policy is not None:
            policy.to(CONFIG.INFER_DEVICE)

        z = np.array([0.3, 0, 0])

        prev_frame_draw = time.time() - t
        prev_step_time = self.data.time

        while self.display.running:
            if policy is not None:
                # concatenated = np.concatenate([obs, z])
                # tensor = torch.from_numpy(concatenated).to(torch.float32)
                obs_tensor = torch.from_numpy(obs).to(torch.float32).to(CONFIG.INFER_DEVICE)
                z_tensor = torch.from_numpy(z).to(torch.float32).to(CONFIG.INFER_DEVICE)

                with torch.no_grad():
                    # mean, std, _ = policy(tensor)
                    mean, std, _ = policy(obs_tensor, z_tensor)

                dist = Normal(mean, std)
                action = dist.sample()

            else:
                action = torch.as_tensor([0.1,0.1,0.1,1,1,1,0.5,0.5,0.5,0.5,0.5,0.5])
                # action = torch.as_tensor([0,0,0,0,0,0,0,0,0,0,0,0])
                # action = torch.as_tensor([])
                # action = torch.as_tensor([
                    # 0.5, 0.5, 0.5, 0.5,
                    # 1, 1, 1, 1,
                    # 0.8, 0.8, 0.8, 0.8
                    # 0.5, 0.5, 0.5, 0.5,
                    # 0.5, 0.5, 0.5, 0.5,
                    # 1, 1, 1, 1
                    # 0.5, 0.5, 0.5, 0.5,
                    # 0.35, 0.35, 0.35, 0.35,
                    # self.data.time*0.3,
                    # self.data.time*0.3,
                    # self.data.time*0.3,
                    # self.data.time*0.3
                # ])

            if not self._stop_step(prev_step_time):
                obs, reward, done, _ = self.step(action)

            if self._display_next_frame(prev_frame_draw, t=t):
                # self.renderer.update_scene(self.data, camera="side")
                self.renderer.update_scene(self.data)
                img = self.renderer.render()

                self.display.draw_img(img)
                prev_frame_draw = time.time()
                prev_step_time = self.data.time

            self.display.handle_close()
            
            if self.data.time > 3:
                self.display.quit()

    def _display_next_frame(self, prev_frame, t=1/60):
        return time.time() >= prev_frame + t

    def _stop_step(self, prev_step_time, t=1/60):
        return self.data.time >= prev_step_time + t

    def _initialise_display(self):
        self.renderer = mujoco.Renderer(self.model, width=CONFIG.DISPLAY_W, height=CONFIG.DISPLAY_H)
        self.display = Display(W=CONFIG.DISPLAY_W, H=CONFIG.DISPLAY_H)
