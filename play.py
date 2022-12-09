import argparse
from gui import *


def main():

    parser = arg_parser()
    args, unknown = parser.parse_known_args()

    start(args.model_path, WIDTH=500)


def arg_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='model/',
                        help="Location of the GPT2 model.")

    return parser



if __name__ == '__main__':

    main()
