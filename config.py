import argparse


def parse_args():
    """
    parsing and configuration
    :return: parse_args
    """
    desc = "Tensorflow implementation of ssd-tensorflow"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--GPU', type=str, default='0',
                        help='The number of GPU used')
    parser.add_argument('--module', type=str, default='test',
                        help='Module to select: train, test ...')
    parser.add_argument('--dataset', type=str, default='data/VOC2012-train',
                        help='Path of the dataset')
    parser.add_argument('--tfrecord_dir', type=str, default='data/VOC-tfrecord',
                        help='Path of the dataset transferred to tfrecord')
    parser.add_argument('--split', type=str, default='train',
                        help='Type of data to transfer: train, test ...')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='The size of each training batch')
    parser.add_argument('--num_epochs', type=int, default=2,
                        help='The number of the total training epoch')

    return parser.parse_args()
