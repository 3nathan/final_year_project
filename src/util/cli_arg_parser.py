import argparse
from pathlib import Path

def parse():
    parser = argparse.ArgumentParser(description="A MuJoCo reinforcement learning program")

    parser.add_argument('-m', '--model',
                        type=Path,
                        help="path to the MuJoCo XML model file")

    # need to implement once ML models are implemented
    parser.add_argument('-t', '--train',
                        action='store_true',
                        help="train model")

    parser.add_argument('-e', '--episodes',
                        type=int,
                        help="number of episodes in the training loop")

    # # need to implement once ML models are implemented
    # parser.add_argument('-c', '--control',
    #                     type=str,
    #                     help="Path to load ML model control file")

    # # need to implement image mode
    # parser.add_argument('-v', '--video',
    #                     type=str,
    #                     help="Render video or image from the simulation")

    # # need to implement image mode functionality
    # parser.add_argument('-T', '--time',
    #                     type=int,
    #                     help="Duration of video or time of screenshot simulation render")

    # # need to implement
    # parser.add_argument('-C', '--camera',
    #                     type=str,
    #                     help="Camera to use in the simulation render")

    args = parser.parse_args()

    return args
