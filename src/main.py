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
from learning.learning import ReinforcementLearning

def main():
    args = parse()

    if args.model:
        if args.model.is_file():
            print(f"Using MuJoCo model: {args.model.resolve()}")
        else:
            print(f"Error: file not found at {args.model}")
            print("Exiting...")
            exit(1)
    else:
        print("No MuJoCo model provided")
        print("Exiting...")
        exit(1)

    # if args.model == None:
    #     print("No physics model provided")
    #     print("Exiting...")
    #     exit(0)

    # simulation = Sim(args.model)

    # if args.video == "video":
    #     frames = simulation.render_video(duration=args.time)
    #     display = Display(len(frames[0][0]), len(frames[0]))
    #     display.show_video(frames)

    # elif args.video == "image":
    #     img = simulation.render_image(time=args.time)
    #     display = Display(len(img[0]), len(img))
    #     display.show_img(img)

    if args.train == True:
        print("Training RL model:")
        training = ReinforcementLearning(str(args.model))
        training.train()
        

if __name__ == '__main__':
    main()
