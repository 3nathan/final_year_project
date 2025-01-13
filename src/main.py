#!/Users/nathan/imperial/fourth_year/fyp/venv/bin/python3
# utilities
import sys
import argparse

# simulation modules
from sim.sim import Sim
from sim.display import Display

def main():
    parser = argparse.ArgumentParser(description="A MuJoCo reinforcement learning program")

    parser.add_argument('-m', '--model',
                        type=str,
                        help="Path to the MuJoCo XML model file")

    # need to implement once ML models are implemented
    parser.add_argument('-t', '--train',
                        type=bool,
                        help="Train control model")

    # need to implement once ML models are implemented
    parser.add_argument('-c', '--control',
                        type=str,
                        help="Path to load ML model control file")

    # need to implement image mode
    parser.add_argument('-v', '--video',
                        type=str,
                        help="Render video or image from the simulation")

    # need to implement image mode functionality
    parser.add_argument('-T', '--time',
                        type=int,
                        help="Duration of video or time of screenshot simulation render")

    # need to implement
    parser.add_argument('-C', '--camera',
                        type=str,
                        help="Camera to use in the simulation render")

    args = parser.parse_args()

    if args.model == None:
        print("No physics model provided")
        print("Exiting...")
        exit(0)

    simulation = Sim(args.model)

    if args.video == "video":
        frames = simulation.render_video(duration=args.time)
        display = Display(len(frames[0][0]), len(frames[0]))
        display.show_video(frames)

    elif args.video == "image":
        img = simulation.render_image(time=args.time)
        display = Display(len(img[0]), len(img))
        display.show_img(img)

if __name__ == '__main__':
    main()
