import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
from torch.distributions import Normal

from sim.sim_env import SimEnv
from learning.models import GaitPolicy

from util.config import Config
from util.functions import Logger

CONFIG = Config()

LR = 1e-2

# using:
# https://pytorch.org/tutorials/intermediate/reinforcement_ppo.html#value-network

# most of this is using this code:
# https://www.geeksforgeeks.org/reinforcement-learning-using-pytorch/

class ReinforcementLearning():
    def __init__(self, model_path, policy=GaitPolicy, device=CONFIG.DEVICE, seed=None, save=False, video=False):
        self.env = SimEnv(model_path=model_path, seed=seed, video=video)
        self.save = save

        obs_dim, action_dim = self.env.get_dims()
        self._latent = self._generate_latent_dist(mu=0, sigma=0.2, size=CONFIG.GENGHIS_CTRL_DIM)
        hidden_dims = (512, 512, 512)
        self.policy = policy(obs_dim=obs_dim, action_dim=action_dim, latent_dim=CONFIG.GENGHIS_CTRL_DIM, hidden_dims=hidden_dims)

        self.policy.to(device)
        print(f"Training on {device}")

    def train(self, optimiser=None, episodes=None):
        if optimiser is None:
            optimiser = optim.Adam(self.policy.parameters(), lr=LR)

        if episodes is None:
            episodes = 1000

        # TODO:
        # perform inference on cpu if using a mac
        # avoids costs associated with sim env data transfer latency

        # if CONFIG.USING_MAC is True:
        #     device_infer = torch.device("cpu")
        # else:
        #     device_infer = CONFIG.DEVICE
        # device_train = CONFIG.DEVICE

        if self.save:
            print(f"Saving weights to {CONFIG.MODELS_PATH}/neural_networks/genghis.pth")

        print(f"Beginning training on {episodes} episodes")

        logger = Logger(["Episode", "Reward", "Loss"])

        for episode in range(episodes):
            z = self._sample_latent_dist()
            # hard coded for genghis reward function
            self.env.latent_velocity = z
            # hard coded for genghis reward function

            log_probs = []      # TODO: check what this corresponds to
            rewards = []
            
            observation = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                input_tensor = self._get_input_tensor(observation, z)
                mean, std = self.policy(input_tensor)
                # print(f"Mean:\n {mean}")
                # print(f"std:\n {std}")
                distribution = Normal(mean, std)
                action = distribution.sample()
                log_prob = distribution.log_prob(action).sum()

                observation, reward, done, _ = self.env.step(action)
                
                log_probs.append(log_prob)
                rewards.append(reward)
                total_reward += reward

            returns = self._compute_discounted_rewards(rewards)
            # returns = (returns - returns.mean()) / (returns.std() + 1e-8)   # normalise

            policy_loss = -torch.stack(log_probs) * returns
            policy_loss = policy_loss.sum()

            optimiser.zero_grad()
            policy_loss.backward()
            optimiser.step()

            if episode % CONFIG.LOG_INTERVAL == 0:
                logger.log([episode, sum(rewards), policy_loss.item()])

                if self.save:
                    torch.save(self.policy.state_dict(), CONFIG.MODELS_PATH+'/neural_networks/genghis.pth')

    def _get_input_tensor(*vectors, device=CONFIG.DEVICE, dtype=torch.float32):
        concatenated = np.concatenate(vectors[1:])
        tensor = torch.from_numpy(concatenated).to(dtype)
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
        # return np.zeros(2)
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
