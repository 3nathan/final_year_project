import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F

from sim.sim_env import SimEnv
from learning.models import GaitPolicy

# using:
# https://pytorch.org/tutorials/intermediate/reinforcement_ppo.html#value-network

# most of this is using this code:
# https://www.geeksforgeeks.org/reinforcement-learning-using-pytorch/

class ReinforcementLearning():
    # TODO: prepend the model path with {$BASE} when the functionality is implemented
    def __init__(self, model_path="models/physics/genghis.xml", policy=GaitPolicy, seed=None):
        self.env = SimEnv(model_path=model_path, seed=seed)

        obs_dim, action_dim = self.env.get_dims()
        latent_dim, hidden_dims = 4, (512, 512, 512)

        self.policy = policy(obs_dim=obs_dim, action_dim=action_dim, latent_dim=latent_dim)

        self.latent_dist = self._generate_latent_dist(mu=0, sigma=4, size=latent_dim)

    # optimizer = optim.Adam(policy.parameters(), lr=1e-2)
    def train(self, optimiser, episodes=1000):
        for episodes in range(episodes):
            z = _sample_latent_dist()
            observation = self.env.reset()
            log_probs = []      # TODO: check what this corresponds to
            rewards = []
            done = False

            while not done:
                distribution = self.policy(observation, z)
                m = Categorical(distribution)       # TODO: need to include this 'Categorical' class
                action = distribution.sample()
                observation, reward, done, _ = self.env.step(action)

                log_probs.append(m.log_prob(action))
                rewards.append(reward)

                if done:
                    episode_rewards.append(sum(rewards))
                    discounted_rewards = compute_discounted_rewards(rewards)    # TODO: need to check if this needs to be implemented
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

    def _generate_latent_dist(self, mu, sigma, size):
        return np.random.normal(mu, sigma, size)

    def _sample_latent_dist(self):
        return self.latent_dist.sample
