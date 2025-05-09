import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Config():
    def __init__(self):
        # RL configs
        self.EPISODE_DURATION = 10

        # Genghis configs
        self.GENGHIS_CTRL_DIM = 4
