# Requierments for cluster runs to be combined with installation of the necessary modules in slurm;
from __future__ import print_function

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


import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import codecs
import scipy.misc as m
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
import torchvision
from torch.utils.data import Subset
from torchvision import transforms
from torch.autograd import Variable

from enum import Enum, IntEnum, unique

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# See PyCharm help at https://www.jetbrains.com/help/pycharm/