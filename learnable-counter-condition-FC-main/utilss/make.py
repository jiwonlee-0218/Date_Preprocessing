import os
import json
import csv
import numpy as np
import torch


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def mk_paths(args, f, main_path):

    model_path = main_path + 'ckpt/cv{}/'.format(f)
    create_dir(model_path)
    paths = [main_path, model_path]

    return paths

