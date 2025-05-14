import torch
import platform

class Config():
    def __init__(self):
        # RL configs
        self.DEVICE = torch.device(
            # "mps" if torch.backends.mps.is_available() else
            # "cuda" if torch.cuda.is_available() else
            "cpu"
        )
        self.EPISODE_DURATION = 10
        self.LOG_INTERVAL = 5
        self.MODELS_PATH = "/Users/nathan/imperial/fourth_year/fyp/codebase/models"

        # Machine configs
        self.USING_MAC = platform.system() == "Darwin"

        # Genghis configs
        self.GENGHIS_CTRL_DIM = 2

        # Display configs
        self.DISPLAY_W = 640
        self.DISPLAY_H = 480
