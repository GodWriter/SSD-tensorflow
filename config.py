import argparse


def parse_args():
    """
    parsing and configuration
    :return: parse_args
    """
    desc = "Tensorflow implementation of ssd-tensorflow"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--module', type=str, default='test',
                        help='Module to select: train, test ...')

    return parser.parse_args()
