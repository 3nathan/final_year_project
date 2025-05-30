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

    parser.add_argument('-a', '--algorithm',
                        type=str,
                        help="RL algorithm to use in training")

    parser.add_argument('-s', '--save',
                        action='store_true',
                        help="save neural network weights")

    parser.add_argument('-n', '--neural_network',
                        type=Path,
                        help="path to load neural network from")

    parser.add_argument('-tr', '--trajectories',
                        type=int,
                        help="number of trajectories in the training loop")

    parser.add_argument('-b', '--batch_size',
                        type=int,
                        help="size of batch collected per weight update")

    parser.add_argument('-e', '--epochs',
                        type=int,
                        help="number of epochs per batch of training data")

    parser.add_argument('-v', '--video',
                        action='store_true',
                        help="render video of simulation")

    args = parser.parse_args()

    check_syntax(args)

    return args

def check_syntax(args):
    if args.model:
        if not args.model.is_file():
            print(f"Error: file not found at {args.model}")
            print("Exiting...")
            exit(1)

    else:
        print("No MuJoCo model provided")
        print("Exiting...")
        exit(1)

    if args.neural_network:
        if not args.neural_network.is_file():
            print(f"Error: file not found at {args.model}")
            print("Exiting...")
            exit(1)

    if args.algorithm is not None and args.train is True:
        algorithms = ["policy_gradient", "PPO"]

        if args.algorithm not in algorithms:
            print(f"Error: {args.algorithm} is not an accepted algorithms")
            print("Accepted algorithms:")

            for algorithm in algorithms:
                print(algorithm)

            print("Exiting...")
            exit(1)

    if args.batch_size is not None and args.train is True:
        if args.batch_size <= 0:
            print("Error: negative or zero batch size not allowed")
            print("Exiting...")
            exit(1)

    if args.epochs is not None and args.train is True:
        if args.epochs <= 0:
            print("Error: negative or zero number of epochs not allowed")
            print("Exiting...")
            exit(1)
