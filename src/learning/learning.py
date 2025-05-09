import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
from torch.distributions import Categorical

from sim.sim_env import SimEnv
from learning.models import GaitPolicy

from util.config import Config
from util.config import DEVICE

CONFIG = Config()

LR = 1e-2

# using:
# https://pytorch.org/tutorials/intermediate/reinforcement_ppo.html#value-network

# most of this is using this code:
# https://www.geeksforgeeks.org/reinforcement-learning-using-pytorch/

class ReinforcementLearning():
    def __init__(self, model_path, policy=GaitPolicy, seed=None):
        self.env = SimEnv(model_path=model_path, seed=seed)

        obs_dim, action_dim = self.env.get_dims()
        self._latent = self._generate_latent_dist(mu=0, sigma=1, size=CONFIG.GENGHIS_CTRL_DIM)
        hidden_dims = (512, 512, 512)
        self.policy = policy(obs_dim=obs_dim, action_dim=action_dim, latent_dim=CONFIG.GENGHIS_CTRL_DIM, hidden_dims=hidden_dims)

        self.policy.to(DEVICE)

    def train(self, optimiser=None, episodes=1000):
        if optimiser is None:
            optimiser = optim.Adam(self.policy.parameters(), lr=LR)

        print(f"Beginning training on {episodes} episodes")

        for episode in range(episodes):
            z = self._sample_latent_dist()
            observation = self.env.reset()
            log_probs = []      # TODO: check what this corresponds to
            rewards = []
            done = False

            while not done:
                input_tensor = self._get_input_tensor(observation, z)
                distribution = self.policy(input_tensor)
                print("Exiting program from the training loop, further debugging needed")
                exit(0)
                m = Categorical(distribution)
                action = distribution.sample()
                observation, reward, done, _ = self.env.step(action)

                log_probs.append(m.log_prob(action))
                rewards.append(reward)
                exit()

                if done:
                    episode_rewards.append(sum(rewards))
                    discounted_rewards = self,_compute_discounted_rewards(rewards)    # TODO: need to check if this needs to be implemented
                    # TODO: need to alter the following depending on the rl algorithm used
                    policy_loss = []
                    for log_prob, Gt in zip(log_probs, dicounted_rewards):
                        policy_loss.append(-log_prob * Gt)
                    optimiser.zero_grad()
                    policy_loss = torch.cat(policy_loss).sum()
                    policy_loss.backward()
                    optimiser.step()

                    if episode % 50 == 0:
                        print(f"Episode {episode}, Total Reward: {sum(rewards)}")
                    break

    def _get_input_tensor(*vectors, device=DEVICE, dtype=torch.float32):
        concatenated = np.concatenate(vectors[1:])
        tensor = torch.from_numpy(concatenated).to(dtype)

        if device is not None:
            tensor = tensor.to(device)

        return tensor

    # TODO: change this
    #       should generate x, xdot, theta, thetadot, etc. params
    def _generate_latent_dist(self, mu, sigma, size):
        latent = {
            'means':    np.full(size, mu),
            'std':      np.full(size, sigma)
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

        discounted_rewards = torch.tensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-5)
        
        return discounted_rewards
