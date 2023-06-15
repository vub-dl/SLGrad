# Requierments for cluster runs to be combined with installation of the necessary modules in slurm;

# Basics
import os
import sys
import gc
import math
import random as rn
import random
import argparse

#numpy
import numpy as np
from scipy.optimize import minimize

#pandas
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.base import clone
from itertools import islice, chain, repeat


import wandb
import fnmatch

# Pytorch
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, TensorDataset
from torch import nn
import torch.utils.data
from torch.hub import load_state_dict_from_url
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F
import json

from enum import Enum, IntEnum, unique

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
