import argparse
from pathlib import Path

def parse():
    parser = argparse.ArgumentParser(description="A MuJoCo reinforcement learning program")

    parser.add_argument('-m', '--model',
                        type=Path,
                        help="path to the MuJoCo XML model file")

    parser.add_argument('-t', '--train',
                        action='store_true',
                        help="train neural network")

    parser.add_argument('-s', '--save',
                        action='store_true',
                        help="save neural network weights")

    parser.add_argument('-e', '--episodes',
                        type=int,
                        help="number of episodes in the training loop")

    parser.add_argument('-v', '--video',
                        action='store_true',
                        help="render video of simulation")

    args = parser.parse_args()

    return args
