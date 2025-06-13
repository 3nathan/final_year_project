# #!/Users/nathan/imperial/fourth_year/fyp/codebase/venv/bin/python3
# utilities
import sys
import argparse
from util.cli_arg_parser import parse
from pathlib import Path

# simulation modules
from sim.sim_env import SimEnv

# machine learning modules
from learning.models_copy import ConnectPolicy
from learning.models_copy import SplitPolicy
from learning.models_copy import CommPolicy
from learning.models_copy import EncodePolicy
from learning.models_copy import LoadAblate
# from learning.training import ReinforcementLearning
from learning.training_copy import ReinforcementLearning

from util.config import Config

CONFIG = Config()

def main():
    args = parse()

    if args.train is True:
        print("Training RL model")

        if args.neural_network == 'connect':
            policy = ConnectPolicy
        elif args.neural_network == 'split':
            policy = SplitPolicy
        elif args.neural_network == 'comm':
            policy = CommPolicy
        elif args.neural_network == 'encode':
            policy = EncodePolicy
        elif args.neural_network == 'ablate':
            policy = LoadAblate
        else:
            policy = ConnectPolicy

        model = ReinforcementLearning(str(args.model), policy=policy, save=args.save, video=args.video)
        model.train(algorithm=args.algorithm, trajectories=args.trajectories, batch_size=args.batch_size, epochs=args.epochs)

    elif args.video is True:
        env = SimEnv(str(args.model), video=True)

        print("Demoing simulation")
        env.run_demo()

if __name__ == '__main__':
    main()
