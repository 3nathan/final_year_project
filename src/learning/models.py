import torch
import torch.nn as nn
from torch.distributions import Normal

from util.config import Config

CONFIG = Config()

def construct_linear(input_dim, hidden_dims):
    layers = []
    last_dim = input_dim
    for curr_dim in hidden_dims:
        layers.append(nn.Linear(last_dim, curr_dim))
        layers.append(nn.LayerNorm(curr_dim))
        layers.append(nn.LeakyReLU(0.1))
        last_dim = curr_dim

    return layers

# this is a latent-conditioned RL policy
# the latent variable z represents gaits
#   this could be:
#       body pose
#       xdot wrt facing direction
#       thetadot
# it outputs an action distribution (joint torgues/target positions)

# what is the follower reward function?
# Option 1: shared task reward (cooperation)
#       both agents sharing the same communicated goal (may not adhere to project - communication is limited)
# Option 2: Auxiliary consistence reward
#       reward follower for producing actions or behaviours that are consistent with the leaders intent
#       reward reconstruction of leader intent
#       r_follower += -||predicted_behaviour_from_z - actual_behaviour||
# Option 3: reward adherence to gait control vector
#       reward follower for matching gait features and or predicted dynamics


class GaitPolicy(nn.Module):
    def __init__(self, obs_dim, latent_dim, action_dim, hidden_dims=(512, 512, 512), activation=nn.LeakyReLU):
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        input_dim = obs_dim + latent_dim

        # setup network based on given hidden dimensions
        layers = construct_linear(input_dim, hidden_dims)
        self.model = nn.Sequential(*layers)

        self.mean_head = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1], action_dim)

    def forward(self, x):
        obs = x[:self.obs_dim]
        z = x[-self.latent_dim:]
        x = self.model(x)

        mean = torch.sigmoid(self.mean_head(x))
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = torch.exp(log_std)

        return mean, std, x

    def save_weights(self, path=str(CONFIG.ROOT)+"/models/neural_networks/quadruped.pth"):
        torch.save(self.state_dict(), path)

    def load_weights(self, path=str(CONFIG.ROOT)+"/models/neural_networks/quadruped.pth"):
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        self.load_state_dict(state_dict)

class DisconnectedPolicy(nn.Module):
    def __init__(self, obs_dim, latent_dim, action_dim, hidden_dims=(512, 512, 512), activation=nn.LeakyReLU):
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim

        self.left = GaitPolicy(obs_dim, latent_dim, action_dim//2, hidden_dims) 
        self.right = GaitPolicy(obs_dim, 0, action_dim//2, hidden_dims) 

    def forward(self, x):
        obs = x[:self.obs_dim]
        z = x[-self.latent_dim:]

        left_mean, left_std, _ = self.left(x)
        right_mean, right_std, _ = self.right(obs)

        mean = torch.stack((left_mean, right_mean), dim=1)
        std = torch.stack((left_std, right_std), dim=1)

        return mean, std, _

class DisconnectedCommPolicy(nn.Module):
    def __init__(self, obs_dim, latent_dim, action_dim, hidden_dims=(512, 512, 512), activation=nn.LeakyReLU):
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim

        self.left = GaitPolicy(obs_dim, latent_dim, action_dim//2, hidden_dims) 
        self.right = GaitPolicy(obs_dim, latent_dim, action_dim//2, hidden_dims) 

    def forward(self, x):
        left_mean, left_std, _ = self.left(x)
        right_mean, right_std, _ = self.right(x)

        mean = torch.stack((left_mean, right_mean), dim=1)
        std = torch.stack((left_std, right_std), dim=1)

        return mean, std, _

class EncodedCommPolicy(nn.Module):
    def __init__(self, obs_dim, latent_dim, action_dim, hidden_dims=(512, 512, 512), encoded_dims=(256, 128, 1), activation=nn.LeakyReLU):
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim

        self.left = GaitPolicy(obs_dim, latent_dim, action_dim//2, hidden_dims) 
        self.right = GaitPolicy(obs_dim, encoded_dim, action_dim//2, hidden_dims) 

        layers = construct_linear(hidden_dims[-1], encoded_dims)
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        obs = x[:self.obs_dim]

        left_mean, left_std, high_lvl_features = self.left(x)

        encoded = self.encoder(high_lvl_features)
        right_mean, right_std, _ = self.right(torch.stack((obs, encoded), dim=1)

        mean = torch.stack((left_mean, right_mean), dim=1)
        std = torch.stack((left_std, right_std), dim=1)

        return mean, std, _

# this is a variational encoder used for multi-agent gait coordination in RL
# it encodes input features into a latent variable z for inter-agent communication
#   z captures gait and policy-relevant information under bandwidth constraints
# the encoder outputs a latent distribution (mean, std), from which z is sampled or derived
# z is used by a downstream policy network and shared with another agent
# optionally:
#   - a decoder is used to reconstruct input or predict auxiliary signals
#   - training may include reconstruction loss, KL regularization, and RL reward

# an option for the input features are:
# 1: VAE input = policy hidden layer
#       this encodes processed features that drive the 'leader's' decision
# 2: VAE input = [state + action]
#       skips the processed features and transmits the infor the 'leader' has without the high level
#       features that have been computed in the policy hidden layer

# class VAE(nn.Module):
#     def __init__(self, input_dim, output_dim, latent_dim, hidden_dims=(256, 128), activation=nn.LeakyReLU):
#         super().__init__()
#         encoder_layers = construct_linear(input_dim, hidden_dims)
#         decoder_layers = construct_linear(latent_dim, reversed(hidden_dims) + (output_dim,))

#         self.encoder = nn.Sequential(*layers)
#         self.decoder = nn.Sequential(*layers)

#         self.mean_head = nn.Linear(hidden_dims[-1], latent_dim)
#         self.log_var_head = nn.Linear(hidden_dims[-1], latent_dim)

#     def encode(self, x):
#         x = self.encoder(x)
        
#         mean = self.mean_head(x)
#         log_var = self.log_var_head(x)
#         # log_var = torch.clamp(log_var, min=-20, max=2)

#         return mean, log_var

#     def reparameterise(self, mean, log_var):
#         std = torch.exp(0.5*log_var)
#         return Normal(mean, std).sample()

#     def decode(self, z):
#         return self.decoder(z)

#     def forward(self, x):
#         mean, log_var = self.encode(x)
#         z = self.reparameterise(mean, log_var)
#         recon = self.decoder(z)

#         return recon, mean, log_var

    # TODO: investigate whether using deterministic is better
    #       use deterministic enables outputing the latent vector as the mean and disables
    #       outputting the latent vector as a sample of a normal distribution
