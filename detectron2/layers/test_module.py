from __future__ import print_function

import argparse

import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torchvision import datasets, transforms

from group_norm  import *

from torch.utils.data import DataLoader, Dataset, TensorDataset

from gamma_correction import *

from Clustering import *

import random

import os

import numpy as np

from GN_w_BN import GN_w_BN_f

tmp = np.zeros((1, 10, 3, 3))
for i in range(10):
    tmp[0, i, :, :] = i * np.ones((3, 3))
t = GN_w_BN_f(tmp, 5)
t
