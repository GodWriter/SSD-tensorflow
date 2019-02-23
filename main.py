import os

from config import parse_args
from solver import Solver
from dataloader import Dataset

args = parse_args()

# solver = Solver(args)
dataset = Dataset(args)

if __name__ == '__main__':
    module = args.module

    os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU

    if module == 'create_dataset':
        dataset.create_dataset()
    elif module == 'test_dataset':
        dataset.test_dataset()
    else:
        print("This module has not been created!")
