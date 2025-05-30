import torch
import torch.nn as nn
from torch.distributions import Normal

from util.config import Config

CONFIG = Config()

# this is a latent-conditioned RL policy
# the latent variable z represents gaits
#   this could be:
#       body pose
#       xdot wrt facing direction
#       thetadot
# it outputs an action distribution (joint torgues/target positions)

def construct_linear(input_dim, hidden_dims):
    layers = []
    last_dim = input_dim
    for curr_dim in hidden_dims:
        layers.append(nn.Linear(last_dim, curr_dim))
        layers.append(nn.LayerNorm(curr_dim))
        layers.append(nn.LeakyReLU(0.1))
        last_dim = curr_dim

    return layers

class GaitPolicy(nn.Module):
    def __init__(self, obs_dim, latent_dim, action_dim, hidden_dims=(512, 512, 512), activation=nn.LeakyReLU):
        super().__init__()
        input_dim = obs_dim + latent_dim

        # setup network based on given hidden dimensions
        layers = construct_linear(input_dim, hidden_dims)
        self.model = nn.Sequential(*layers)

        self.mean_head = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1], action_dim)

    def forward(self, x):
        x = self.model(x)

        mean = torch.sigmoid(self.mean_head(x))
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = torch.exp(log_std)

        return mean, std

    def save_weights(self, path=str(CONFIG.ROOT)+"/models/neural_networks/quadruped.pth"):
        torch.save(self.state_dict(), path)

    def load_weights(self, path=str(CONFIG.ROOT)+"/models/neural_networks/quadruped.pth"):
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        self.load_state_dict(state_dict)

class VAE(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=(256, 128), latent_dim, activation=nn.LeakyReLU):
        super().__init__()
        
        encoder_layers = construct_linear(input_dim, hidden_dims)
        decoder_layers = construct_linear(latent_dim, reversed(hidden_dims))

        self.encoder = nn.Sequential(*layers)
        self.decoder = nn.Sequential(*layers)

        self.mean_head = nn.Linear(hidden_dims[-1], latent_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1]. latent_dim)

        self.output = nn.Linear(hidden_dims[0], output_dim)

    def forward(self, x):
        x = self.encoder(x)

        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = torch.exp(log_std)

        sample = Normal(mean, std).sample()
        output = self.decoder(sample)

        return output, mean, std
