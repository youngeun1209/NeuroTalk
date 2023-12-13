#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 22:02:38 2022

@author: yelee
"""

from myDataset_makeMel import myDataset
import torch
from librosa.util import normalize
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn
import librosa.display
from librosa.util import nnls
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import pandas as pd
import os
import torchaudio
import noisereduce as nr
from utils import audio_denorm

dataname = 'CSV0_CSPim/'

fileDir = './data_dir/' + dataname
imgDir = './data_dir/' + dataname

subjects = ['sub1', 'sub2', 'sub3', 'sub4', 'sub5', 'sub6'];

   
for subNum in range(1):
    task = 'Voice'  # ImaginedEEG, OvertEEG, OvertEEG_raw
    recon = 'Voice'
    
    sub = subjects[subNum]
    
    datadir = fileDir + sub
    classes= 13 #13
    num_workers = 4
    
    GPU_NUM = 1
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    
    torch.cuda.set_device(device) # change allocation of current GPU
    print ('Current cuda device ', torch.cuda.current_device()) # check
    print(torch.cuda.device_count())
    
    trainset = myDataset(mode=0, data=datadir,task=task,recon=recon)  # file='./EEG_EC_Data_csv/train.txt'
    train_loader = DataLoader(
        trainset, batch_size=780, shuffle=False, num_workers=num_workers, pin_memory=True) # 780
    
    
    testset = myDataset(mode=1, data=datadir,task=task,recon=recon)  
    test_loader = DataLoader(
        testset, batch_size=260, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    valset = myDataset(mode=2, data=datadir,task=task,recon=recon)
    val_loader = DataLoader(
        valset, batch_size=260, shuffle=False, num_workers=num_workers, pin_memory=True)
    
        
    #%% voice data csv save
    loader = [val_loader, test_loader, train_loader]
    naming = ['val','ts','tr']
    foldername = ['val', 'test','train']
    
    for kk in range(len(loader)):
        total_batches = len(loader[kk])
        signal, _, target_cl, _, max_num = next(iter(loader[kk]))
    
        save_dire = imgDir + sub + '/' + foldername[kk] + '/OvertVoice_mel/'

        for i in range(len(signal)):
            y_voice = signal[i,0].cpu()
            sample_rate_ori = 8000
            sample_rate = 22050
            
            # Resampling
            y_voice = torchaudio.functional.resample(y_voice, sample_rate_ori, sample_rate)   
            
            # Noise reduction
            y_voice_nr = nr.reduce_noise(y=y_voice, sr=sample_rate)
            
            # sd.play(y_voice, sample_rate)
            # sd.play(y_voice_nr, sample_rate_ori)
    
            y_voice_nr = torch.Tensor(y_voice_nr)
            
            n_fft = 1024
            win_length = 1024
            hop_length = win_length//4
            n_mel_channels = 80 #64
    
            mel_fmin=0.0
            mel_fmax=8000.0 #int(sample_rate/2)
    
            hann_window = torch.hann_window(win_length)
    
            # mel_basis = librosa_mel_fn(sample_rate, n_fft=n_fft)
            mel_basis = librosa_mel_fn(sample_rate, n_fft=n_fft,n_mels=n_mel_channels,
                                        fmin=mel_fmin,fmax=mel_fmax)
    
            mel_basis = torch.from_numpy(mel_basis)
    
            p = (n_fft - hop_length) // 2 # voice: 256 , EEG:64 #(n_fft - hop_length) // 2
            y = F.pad(y_voice_nr, (p, p))
    
            spec = torch.stft(y, 
                              n_fft, 
                              hop_length=hop_length,
                              win_length=win_length,
                              window=hann_window,
                              center=False)
                    
            magnitude = torch.sqrt(spec[:,:,0]**2 + spec[:,:,1]**2)
            mel = torch.matmul(mel_basis, magnitude)
            
            mel = torch.log(torch.clamp(mel, min=1e-5))
            
            # fig, ax = plt.subplots()
            # img = librosa.display.specshow(mel.detach().numpy(), x_axis='time',
            #                           y_axis='mel', sr=sample_rate,
            #                           hop_length=hop_length,
            #                           ax=ax)
            # fig.colorbar(img, ax=ax, format='%+2.0f dB')
            # ax.set(title='Mel-frequency spectrogram')

            dataframe = pd.DataFrame(mel)
            dataframe.to_csv(save_dire + 'mel_%s%04d.csv'%(naming[kk],i), header=False, index=False)
            
            
    
    #%% voice data csv save
    loader = [val_loader, test_loader, train_loader]
    naming = ['val','ts','tr']
    foldername = ['val', 'test','train']
    print('sub ' + str(subNum) + ': ' + dataname)
    for kk in range(len(loader)):
        total_batches = len(loader[kk])
        signal, _, target_cl, _, max_num = next(iter(loader[kk]))
    
        save_dire = imgDir + sub + '/' + foldername[kk] + '/OvertVoice_16000/'
        
        # %
        i=0
        for i in range(len(signal)):
            y_voice = signal[i,0].cpu() # 5:FC5 for 30ch, 7 FC5 for 64ch
            # %   
            sample_rate_ori = 8000
            sample_rate = 16000
            
           
            y_voice = torchaudio.functional.resample(y_voice, sample_rate_ori, sample_rate)   
            
            y_voice_nr = nr.reduce_noise(y=y_voice, sr=sample_rate)      
            
            y_voice_nr = audio_denorm(y_voice_nr)
            
            dataframe = pd.DataFrame(y_voice_nr)
            dataframe.to_csv(save_dire + 'voice_%s%04d.csv'%(naming[kk],i), header=False, index=False)   
    
          
    
