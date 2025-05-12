import torch
import torch.nn as nn

from util.config import Config

CONFIG = Config()

# this is a latent-conditioned RL policy
# the latent variable z represents gaits
#   this could be:
#       body pose
#       xdot wrt facing direction
#       thetadot
# it outputs an action distribution (joint torgues/target positions)

class GaitPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, latent_dim, hidden_dims=(512, 512, 512), activation=nn.LeakyReLU(0.01)):
        super().__init__()
        input_dim = obs_dim + latent_dim

        # setup network based on given hidden dimensions
        layers = []
        last_dim = input_dim
        for curr_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, curr_dim))
            layers.append(activation)
            last_dim = curr_dim
        
        self.model = nn.Sequential(*layers)

        self.mean_head = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1], action_dim)

    def forward(self, x):
        x = self.model(x)

        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        std = torch.exp(log_std)

        return mean, std

# most of the following code is lifted from the pytorch reinforcement q learning tutorial:
# # https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

# it references the original q learning paper:
# https://arxiv.org/abs/1312.5602

# replay memory (taken from pytorch tutorial)
# stores transitions that the agent observes
# allows later reuse
# random sampling of this data build up decorrelated batches
# shown that this greatly stabilises and improves DQN training procedure

# TODO: in the future, implement the memory recall functionality for training the DQN model
