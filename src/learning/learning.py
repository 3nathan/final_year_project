import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F

from sim.sim import SimEnv
from learning.models import GaitPolicy

policy = GaitPolicy("""include the arguments for this""")
env = SimEnv("""include the model path""")

# using:
# https://pytorch.org/tutorials/intermediate/reinforcement_ppo.html#value-network

def latent_variable(mu, sigma):
    return np.zeros(4)

def latent_variable_noise(z, noise_scale=0.1):
    z += self.rng.normal(scale=noise_scale, size=z.shape)

def train(env, policy, optimiser, episodes=1000):
    for episodes in range(episodes):
        z = latent_variable()
        observation = env.reset()
        log_probs = []
        rewards = []
        done = False

        while not done:
            distribution = policy(observation, z)
            action = distribution.sample
