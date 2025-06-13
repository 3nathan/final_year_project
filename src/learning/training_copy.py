import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
from torch.distributions import Normal

from sim.sim_env import SimEnv
from learning.models_copy import ConnectPolicy

from util.config import Config
from util.functions import Logger

import sys

CONFIG = Config()

class ReinforcementLearning():
    def __init__(self, model_path, policy=ConnectPolicy, seed=None, save=False, video=False):
        self.env = SimEnv(model_path=model_path, seed=seed, video=video)
        self.save = save
        self.video = video

        obs_dim, action_dim = self.env.get_dims()
        self._latent = self._generate_latent_dist(size=CONFIG.GENGHIS_CTRL_DIM)
        self.policy = policy(obs_dim=obs_dim, latent_dim=CONFIG.GENGHIS_CTRL_DIM, action_dim=action_dim, hidden_dims=CONFIG.HIDDEN_DIMS)

        print(f"Training on {CONFIG.TRAIN_DEVICE}")

    def run_episode(self): # return log_probs, rewards etc
        N = CONFIG.EPISODE_DURATION / self.env.model.opt.timestep
        z_update_interval = np.linspace(0, N, num=CONFIG.Z_UPDATES, endpoint=False, dtype=int)

        log_probs = []
        rewards = []
        states = []
        actions = []
        
        observation = self.env.reset()
        done = False
        episode_reward = 0

        step = 0
        while not done:
            if step in z_update_interval:
                z = self._sample_latent_dist()
                self.env.latent_control = z
            obs_tensor = torch.from_numpy(observation).to(torch.float32).to(CONFIG.INFER_DEVICE)
            z_tensor = torch.from_numpy(z).to(torch.float32).to(CONFIG.INFER_DEVICE)

            # mean, std, _ = self.policy(input_tensor)
            mean, std, _ = self.policy(obs_tensor, z_tensor)
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum()

            observation, reward, done, _ = self.env.step(action.cpu())
            
            log_probs.append(log_prob)
            rewards.append(reward)
            # states.append(input_tensor.cpu())
            states.append((obs_tensor.cpu(), z_tensor.cpu()))
            actions.append(action.cpu())

            episode_reward += reward
            step += 1

        returns = self._compute_discounted_rewards(rewards)

        return log_probs, returns, episode_reward, states, actions

    def train(self, algorithm=None, optimiser=None, trajectories=None, batch_size=None, epochs=None):
        if algorithm is None:
            algorithm = "PPO"
        if optimiser is None:
            optimiser = optim.Adam(self.policy.parameters(), lr=CONFIG.LR)
        if trajectories is None:
            trajectories = CONFIG.TRAJECTORIES
        if batch_size is None:
            batch_size = CONFIG.BATCH_SIZE
        if epochs is None:
            epochs = CONFIG.EPOCHS
        if self.save:
            print(f"Saving weights to {CONFIG.ROOT}/models/neural_networks")

        old_log_probs = None

        print(f"Using {algorithm}")
        print(f"{trajectories} trajectories")
        print(f"{batch_size} episodes per trajectory")
        print(f"{epochs} epochs per training batch")

        logger = Logger(["Trajectory", "Reward"])

        max_reward = 0

        for trajectory in range(trajectories):
            self.policy.to(CONFIG.INFER_DEVICE)

            all_log_probs = []
            all_returns = []
            all_states = []
            all_actions = []
            total_reward = 0

            for _ in range(batch_size):
                log_probs, returns, episode_reward, states, actions = self.run_episode()
                all_log_probs.extend(log_probs)
                all_returns.extend(returns)
                all_states.extend(states)
                all_actions.extend(actions)
                total_reward += episode_reward
            
            self.policy.to(CONFIG.TRAIN_DEVICE)
            
            log_probs_tensor = torch.stack(all_log_probs).to(CONFIG.TRAIN_DEVICE)
            returns_tensor = torch.tensor(all_returns, dtype=torch.float32).to(CONFIG.TRAIN_DEVICE)
            # states_tensor = torch.stack(all_states).to(CONFIG.TRAIN_DEVICE)
            obs_tensor = torch.stack([state[0] for state in all_states]).to(CONFIG.TRAIN_DEVICE)
            z_tensor = torch.stack([state[1] for state in all_states]).to(CONFIG.TRAIN_DEVICE)
            actions_tensor = torch.stack(all_actions).to(CONFIG.TRAIN_DEVICE)

            old_log_probs = log_probs_tensor.detach().clone()

            for epoch in range(epochs):
                mean, std, _ = self.policy(obs_tensor, z_tensor)
                dist = Normal(mean, std)
                new_log_probs = dist.log_prob(actions_tensor).sum(dim=1)

                if algorithm == "policy_gradient":
                    policy_loss = -(new_log_probs * returns_tensor).mean()

                elif algorithm == "PPO":
                    ratio = torch.exp(new_log_probs - old_log_probs)
                    policy_loss = -(torch.minimum(
                        ratio*returns_tensor,
                        torch.clamp(
                            ratio, 
                            1 - CONFIG.EPSILON, 
                            1 + CONFIG.EPSILON
                        )*returns_tensor)
                    ).mean()

                optimiser.zero_grad()
                policy_loss.backward()
                optimiser.step()

            logger.log([trajectory, total_reward / batch_size])

            if self.save and total_reward > max_reward:
                self.policy.save_weights()
                max_reward = total_reward

            if trajectory % 10 == 0 and self.video:
                self.env.run_demo(policy=self.policy)

    def _get_input_tensor(*vectors, device=CONFIG.INFER_DEVICE, dtype=torch.float32):
        concatenated = np.concatenate(vectors[1:])
        tensor = torch.from_numpy(concatenated).to(dtype)

        return tensor

    # TODO: change this
    #       should generate x, xdot, theta, thetadot, etc. params
    #       
    #       at the moment it generates:
    #       [xdot[0], xdot[1], thetadot]
    # def _generate_latent_dist(self, mu, sigma, size):
    def _generate_latent_dist(self, size):
        latent = {
            'means':    np.zeros(size),
            'std':      np.array([0.1, 0.05, 0.1])
        }

        return latent

    def _sample_latent_dist(self):
        return np.random.normal(self._latent["means"], self._latent["std"])

    # TODO: change this when an RL algorithm is decided
    def _compute_discounted_rewards(self, rewards, gamma=0.99):
        discounted_rewards = []
        R = 0

        for r in reversed(rewards):
            R = r + gamma * R
            discounted_rewards.insert(0, R)

        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)
        
        return discounted_rewards
