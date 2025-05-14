# #!/Users/nathan/imperial/fourth_year/fyp/codebase/venv/bin/python3
# utilities
import sys
import argparse
from util.cli_arg_parser import parse
from pathlib import Path

# simulation modules
from sim.sim_env import SimEnv
from sim.display import Display

# machine learning modules
from learning.training import ReinforcementLearning

def main():
    args = parse()

    if args.model:
        if not args.model.is_file():
            print(f"Error: file not found at {args.model}")
            print("Exiting...")
            exit(1)
    else:
        print("No MuJoCo model provided")
        print("Exiting...")
        exit(1)

    if args.train == True:
        print("Training RL model")

        model = ReinforcementLearning(str(args.model), save=args.save, video=args.video)
        model.train(episodes=args.episodes)
        

if __name__ == '__main__':
    main()
