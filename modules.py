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
import torch.nn.functional as F
import torchaudio
import wavio
from utils import data_denorm, mel2wav_vocoder, perform_STT

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

def saveVoice(args, test_loader, models, epoch, losses):
    
    model_g = models[0].eval()
    # model_d = models[1].eval()
    vocoder = models[2].eval()
    model_STT = models[3].eval()
    decoder_STT = models[4]

    input, target, target_cl, voice, data_info = next(iter(test_loader))   
    
    input = input.cuda()
    target = target.cuda()
    voice = torch.squeeze(voice,dim=-1).cuda()
    labels = torch.argmax(target_cl,dim=1)    
    
    with torch.no_grad():
        # run the mdoel
        output = model_g(input)
    
    mel_out = DTW_align(output, target)
    output_denorm = data_denorm(mel_out, data_info[0], data_info[1])
    
    wav_recon = mel2wav_vocoder(torch.unsqueeze(output_denorm[0],dim=0), vocoder, 1)
    wav_recon = torch.reshape(wav_recon, (len(wav_recon),wav_recon.shape[-1]))
    
    wav_recon = torchaudio.functional.resample(wav_recon, args.sample_rate_mel, args.sample_rate_STT)  
    if wav_recon.shape[1] !=  voice.shape[1]:
        p = voice.shape[1] - wav_recon.shape[1]
        p_s = p//2
        p_e = p-p_s
        wav_recon = F.pad(wav_recon, (p_s,p_e))
        
    ##### STT Wav2Vec 2.0
    gt_label = args.word_label[labels[0].item()]
    
    transcript_recon = perform_STT(wav_recon, model_STT, decoder_STT, gt_label, 1)
    
    # save
    wav_recon = wav_recon.cpu().detach().numpy()
    
    str_tar = args.word_label[labels[0].item()].replace("|", ",")
    str_tar = str_tar.replace(" ", ",")
    
    str_pred = transcript_recon[0].replace("|", ",")
    str_pred = str_pred.replace(" ", ",")
    
    title = "Tar_{}-Pred_{}".format(str_tar, str_pred)
    wavio.write(args.savevoice + '/e{}_{}.wav'.format(str(str(epoch)), title), wav_recon, args.sample_rate_STT, sampwidth=1)
        


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


def save_test_all(args, test_loader, models, save_idx=None):
    model_g = models[0].eval()
    # model_d = models[1].eval()
    vocoder = models[2].eval()
    model_STT = models[3].eval()
    decoder_STT = models[4]
    
    save_idx=0
    for i, (input, target, target_cl, voice, data_info) in enumerate(test_loader):
            
        input = input.cuda()
        target = target.cuda()
        voice = torch.squeeze(voice,dim=-1).cuda()
        labels = torch.argmax(target_cl,dim=1)    
        
        with torch.no_grad():
            # run the mdoel
            output = model_g(input)
    
        target = data_denorm(target, data_info[0], data_info[1])
        output = data_denorm(output, data_info[0], data_info[1])
        
        gt_label=[]
        for k in range(len(target)):
            gt_label.append(args.word_label[labels[k].item()])
            
        wav_target = mel2wav_vocoder(target, vocoder, 1)
        wav_recon = mel2wav_vocoder(output, vocoder, 1)
        
        wav_target = torch.reshape(wav_target, (len(wav_target),wav_target.shape[-1]))
        wav_recon = torch.reshape(wav_recon, (len(wav_recon),wav_recon.shape[-1]))
        
        wav_target = torchaudio.functional.resample(wav_target, args.sample_rate_mel, args.sample_rate_STT)  
        wav_recon = torchaudio.functional.resample(wav_recon, args.sample_rate_mel, args.sample_rate_STT)  
        
        if wav_target.shape[1] !=  voice.shape[1]:
            p = voice.shape[1] - wav_target.shape[1]
            p_s = p//2
            p_e = p-p_s
            wav_target = F.pad(wav_target, (p_s,p_e))
            
        if wav_recon.shape[1] !=  voice.shape[1]:
            p = voice.shape[1] - wav_recon.shape[1]
            p_s = p//2
            p_e = p-p_s
            wav_recon = F.pad(wav_recon, (p_s,p_e))
            
        ##### STT Wav2Vec 2.0
        transcript_recon = perform_STT(wav_recon, model_STT, decoder_STT, gt_label, 1)
        
        wav_target = wav_target.cpu().detach().numpy()
        wav_recon = wav_recon.cpu().detach().numpy()
        voice = voice.cpu().detach().numpy()
        
        for batch_idx in range(len(input)):
            
            str_tar = args.word_label[labels[batch_idx].item()].replace("|", ",")
            str_tar = str_tar.replace(" ", ",")
            
            str_pred = transcript_recon[batch_idx].replace("|", ",")
            str_pred = str_pred.replace(" ", ",")

            # Audio save 
            if args.task[0] == 'I':
                title = "Recon_IM_{}-pred_{}".format(str_tar, str_pred)
                wavio.write(args.savevoice + "/" + "%03d_"%(save_idx+1) + title + ".wav", 
                            wav_recon[batch_idx], args.sample_rate_STT, sampwidth=1)
                
            else:
                title = "Recon_SP_{}-pred_{}".format(str_tar, str_pred)
            
                wavio.write(args.savevoice + "/" + "%03d_"%(save_idx+1) + title + ".wav", 
                            wav_recon[batch_idx], args.sample_rate_STT, sampwidth=1)
        
                title = "Target"
                
                wavio.write(args.savevoice + "/" + "%03d_"%(save_idx+1) + title + ".wav", 
                            wav_target[batch_idx], args.sample_rate_STT, sampwidth=1)
                
                title = "Original"
                
                wavio.write(args.savevoice + "/" + "%03d_"%(save_idx+1) + title + ".wav", 
                            voice[batch_idx], args.sample_rate_STT, sampwidth=1)
            save_idx=save_idx+1