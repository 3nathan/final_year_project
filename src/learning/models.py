# most of this code is lifted from the pytorch reinforcement q learning tutorial:
# # https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

# it references the original q learning paper:
# https://arxiv.org/abs/1312.5602

import torch
import torch.nn as nn

# replay memory (taken from pytorch tutorial)
# stores transitions that the agent observes
# allows later reuse
# random sampling of this data build up decorrelated batches
# shown that this greatly stabilises and improves DQN training procedure

# implement this later:
# class ReplayMemory(object):
#     def __init__(self, capacity):
#         self.memory = deque([], maxlen=capacity)

#     def push(self, *args):
#         """Save a transition""")
#         self.memory.append(Transision(*args))

#     def sample(self, batch_size):
#         return random.sample(self.memory, batch_size)

#     def __len__(self):
#         return len(self.memory)

# this is a latent-conditioned RL policy
# the latent variable z represents gaits
#   this could be:
#       body pose
#       xdot wrt facing direction
#       thetadot
# it outputs an action distribution (joint torgues/target positions)

class GaitPolicy(nn.Module):
    def __init__(self, n_observations, n_actions):
        super().__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)
    
    def __init__(self, obs_dim, action_dim, latent_dim, hidden_dims=(256, 256), activation=nn.LeakyReLU(0.01)):
        super().__init__()
        input_dim = obs_dim + latent_dim

        # setup network based on given hidden dimensions
        layers = []
        last_dim = input_dim
        for curr_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, curr_dim))
            layers.append(activation)
            last_dim = curr_dim
        
        model = nn.Sequential(*layers)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs, z):
        x = torch.cat([obs, z], dim=1)
        mean = model(x)
        std = torch.exp(self.log_std)
        return mean, std
