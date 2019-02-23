import os
import tensorflow as tf

from dataloader import Dataset


class Solver(object):
    def __init__(self,
                 args):
        self.args = args
