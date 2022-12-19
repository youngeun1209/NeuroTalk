#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 02:44:56 2022

@author: yelee
"""
import os

import numpy as np
import matplotlib.pyplot as plt
import glob
from torch.nn.utils import weight_norm
import torch
import torch.nn as nn

###################################   DTW    ####################################
def time_warp(costs):
    dtw = np.zeros_like(costs)
    dtw[0,1:] = np.inf
    dtw[1:,0] = np.inf
    eps = 1e-4
    for i in range(1,costs.shape[0]):
        for j in range(1,costs.shape[1]):
            dtw[i,j] = costs[i,j] + min(dtw[i-1,j],dtw[i,j-1],dtw[i-1,j-1])
    return dtw

def align_from_distances(distance_matrix, debug=False):
    # for each position in spectrum 1, returns best match position in spectrum2
    # using monotonic alignment
    dtw = time_warp(distance_matrix)

    i = distance_matrix.shape[0]-1
    j = distance_matrix.shape[1]-1
    results = [0] * distance_matrix.shape[0]
    while i > 0 and j > 0:
        results[i] = j
        i, j = min([(i-1,j),(i,j-1),(i-1,j-1)], key=lambda x: dtw[x[0],x[1]])

    if debug:
        visual = np.zeros_like(dtw)
        visual[range(len(results)),results] = 1
        plt.matshow(visual)
        plt.show()

    return results

def DTW_align(input, target):
    for j in range(len(input)):
        dists = torch.cdist(torch.transpose(input[j],1,0), torch.transpose(target[j],1,0))
        alignment = align_from_distances(dists.T.cpu().detach().numpy())
        input[j,:,:] = input[j,:,alignment]

    return input

#####################################################################################
class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self,yhat,y):
        return torch.sqrt(self.mse(yhat,y))

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        """Given a sequence emission over labels, get the best path string
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          str: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        return "".join([self.labels[i] for i in indices])
    

######################################################################
############                  HiFiGAN                   ##############
######################################################################
def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def apply_weight_norm(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_norm(m)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def save_checkpoint(filepath, obj):
    print("Saving checkpoint to {}".format(filepath))
    torch.save(obj, filepath)
    print("Complete.")


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '????????')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]

######################################################################





