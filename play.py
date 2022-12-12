import argparse
from gui import *


def main():

    parser = arg_parser()
    args, unknown = parser.parse_known_args()

    start(args.model_path, args.beam_width, args.beam_length, WIDTH=500)


def arg_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='model/',
                        help="Location of the GPT2 model.")
    parser.add_argument('--beam_width', type=int, default=3,
                        help="Beam width for inference.")
    parser.add_argument('--beam_length', type=int, default=3,
                        help="Beam length for inference.")

    return parser



if __name__ == '__main__':

    main()
