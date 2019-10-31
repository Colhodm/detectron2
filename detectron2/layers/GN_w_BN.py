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

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def GN_w_BN_f(x, G, eps=1e-5, flag=True):
    x_np = x.detach().numpy()
    #x_np = x
    Nabs, C, H, W = x_np.shape
    if flag:
        N = Nabs
        #x_np_tmp = x.detach().numpy()
        res_x = np.zeros((N, C, H, W))
        x_np_new = np.zeros((C, N * H * W))
        # tmpp = np.zeros((N, H*W))
        for i in range(C):
            x_np_new[i, :] = np.reshape(x_np[:, i, :, :], (1, N * H * W))
        # x_np_new = np.reshape(x_np, (C, N*H*W))
        # x_np = x_np.transpose()
        image_vector = np.asarray(x_np_new)
        Data = data_preparation(n_cluster=G, data=image_vector[:, :])
        for val in range(G):
            inx = np.argwhere(Data.labels_ == val)
            tmp = np.zeros((1, N * H * W * inx.shape[0]))
            for idx, idxx in enumerate(inx):
                tmp[0, idx * N * H * W:(idx + 1) * N * H * W] = x_np_new[idxx[0], :]
            mu = np.mean(tmp)
            sigma = np.std(tmp)
            for idx, idxx in enumerate(inx):
                tmppp = (x_np_new[idxx[0], :] - mu) / np.sqrt(sigma + eps)
                for j in range(N):
                    tmpp = tmppp[j * H * W:(j + 1) * H * W]
                    res_x[j, idxx[0], :, :] = np.reshape(tmpp, (H, W))

    else:
        N = 1
        res_x = np.zeros((Nabs, C, H, W))
        for ii in range(Nabs):
            res_x_tmp = np.zeros((N, C, H, W))
            x_np_new = np.zeros((C, N*H*W))
            #tmpp = np.zeros((N, H*W))
            for i in range(C):
                x_np_new[i, :] = np.reshape(x_np[ii, i, :, :], (1, N*H*W))
            #x_np_new = np.reshape(x_np, (C, N*H*W))
            #x_np = x_np.transpose()
            image_vector = np.asarray(x_np_new)
            Data = data_preparation(n_cluster=G, data=image_vector[:, :])
            for val in range(G):
                inx = np.argwhere(Data.labels_ == val)
                tmp = np.zeros((1, N*H*W*inx.shape[0]))
                for idx, idxx in enumerate(inx):
                    tmp[0, idx*N*H*W:(idx+1)*N*H*W] = x_np_new[idxx[0], :]
                mu = np.mean(tmp)
                sigma = np.std(tmp)
                for idx, idxx in enumerate(inx):
                    tmppp = (x_np_new[idxx[0], :] - mu) / np.sqrt(sigma + eps)
                    for j in range(N):
                        tmpp = tmppp[j*H*W:(j+1)*H*W]
                        res_x_tmp[j, idxx[0], :, :] = np.reshape(tmpp, (H, W))
            res_x[ii, :, :, :] = res_x_tmp
    return torch.from_numpy(res_x)



