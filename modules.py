#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 02:44:56 2022

@author: yelee
"""
import os

import numpy as np
import matplotlib.pyplot as plt
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



def save_checkpoint(state, is_best, save_path, filename):
    """
    Save model checkpoint.
    :param state: model state
    :param is_best: is this checkpoint the best so far?
    :param save_path: the path for saving
    """
    
    torch.save(state, os.path.join(save_path, filename))
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, os.path.join(save_path, 'BEST_' + filename))


def mel2wav_vocoder(mel, vocoder, mini_batch=2):
    waves = []
    for j in range(len(mel)//mini_batch):
        wave_ = vocoder(mel[mini_batch*j:mini_batch*j+mini_batch])
        waves.append(wave_.cpu().detach().numpy())
    wav_recon = torch.Tensor(np.array(waves)).cuda()
    wav_recon = torch.reshape(wav_recon, (len(mel),wav_recon.shape[-1]))
    
    return wav_recon


def perform_STT(wave, model_STT, decoder_STT, gt_label, mini_batch=2):
    # model STT
    emission = []
    with torch.inference_mode():
        for j in range(len(wave)//mini_batch):
            em_, _ = model_STT(wave[mini_batch*j:mini_batch*j+mini_batch])
            emission.append(em_.cpu().detach().numpy())
    emission_recon = torch.Tensor(np.array(emission)).cuda()
    emission_recon = torch.reshape(emission_recon, (len(wave),emission_recon.shape[-2],emission_recon.shape[-1]))
    
    # decoder STT
    transcripts = []
    # corr_num=0
    for j in range(len(wave)):
        transcript = decoder_STT(emission_recon[j])    
        transcripts.append(transcript)
        
    #     if transcript == gt_label[j]:
    #         corr_num = corr_num + 1

    # acc_word = corr_num / len(wave)
        
    return transcripts#, emission_recon, acc_word
