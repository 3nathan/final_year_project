import torch
import platform
from pathlib import Path

class Config():
    def __init__(self):
        # self, general
        self.ROOT = Path(__file__).parent.parent.parent

        # RL configs
        self.TRAIN_DEVICE = torch.device(
            "mps" if torch.backends.mps.is_available() else
            "cuda" if torch.cuda.is_available() else
            "cpu"
        )
        self.INFER_DEVICE = torch.device("cpu")
        self.EPISODE_DURATION = 10
        self.LOG_INTERVAL = 1
        self.TRAJECTORIES = 1000
        self.BATCH_SIZE = 15
        self.EPOCHS = 10
        self.LR = 3e-4  # prev: 3e-4
        self.EPSILON = 0.2
        self.SAVE_INTERVAL = 10
        self.Z_UPDATES = 3
        self.HIDDEN_DIMS = (128, 128)

        # Machine configs
        self.USING_MAC = platform.system() == "Darwin"

        # Genghis configs
        self.GENGHIS_CTRL_DIM = 3

        # Display configs
        self.DISPLAY_W = 640
        self.DISPLAY_H = 480
