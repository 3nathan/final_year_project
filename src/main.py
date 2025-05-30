# #!/Users/nathan/imperial/fourth_year/fyp/codebase/venv/bin/python3
# utilities
import sys
import argparse
from util.cli_arg_parser import parse
from pathlib import Path

# simulation modules
from sim.sim_env import SimEnv

# machine learning modules
from learning.training import ReinforcementLearning
from learning.models import GaitPolicy

from util.config import Config

CONFIG = Config()

def main():
    args = parse()
    if args.train is True:
        print("Training RL model")

        model = ReinforcementLearning(str(args.model), save=args.save, video=args.video)
        model.train(algorithm=args.algorithm, trajectories=args.trajectories, batch_size=args.batch_size, epochs=args.epochs)

    elif args.video is True:
        env = SimEnv(str(args.model), video=True)

        if args.neural_network:
            obs_dim, action_dim = env.get_dims()
            hidden_dims = (512, 512, 512)
            policy = GaitPolicy(obs_dim=obs_dim, action_dim=action_dim, latent_dim=CONFIG.GENGHIS_CTRL_DIM, hidden_dims=hidden_dims)
            policy.load_weights(str(args.neural_network))

        else:
            policy = None

        print("Demoing simulation")

        env.run_demo(policy=policy)

if __name__ == '__main__':
    main()
