
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import glob
from torch.nn.utils import weight_norm

def audio_denorm(data):
    max_audio = 32768.0
    
    data = np.array(data * max_audio).astype(np.float32)
       
    return data


def data_denorm(data, avg, std):
    
    std = std.type(torch.cuda.FloatTensor)
    avg = avg.type(torch.cuda.FloatTensor)
    
    # if std == 0, change to 1.0 for nothing happen
    std = torch.where(std==torch.tensor(0,dtype=torch.float32).cuda(), torch.tensor(1,dtype=torch.float32).cuda(), std)
 
    # change the size of std and avg
    std = torch.permute(std.repeat(data.shape[1],data.shape[2],1),[2,0,1])
    avg = torch.permute(avg.repeat(data.shape[1],data.shape[2],1),[2,0,1])
    
    data = torch.mul(data, std) + avg
       
    return data



def plot_spectrogram(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    fig.canvas.draw()
    plt.close()

    return fig
    
def imgSave(dir, file_name):
    if not os.path.exists(dir):
        os.mkdir(dir)
    plt.tight_layout()
    plt.savefig(dir + file_name)
    plt.clf()


def word_index(word_label, bundle):
    labels_ = ''.join(list(bundle.get_labels()))
    word_indices = np.zeros((len(word_label), 15), dtype=np.int64)
    word_length = np.zeros((len(word_label), ), dtype=np.int64)
    for w in range(len(word_label)):
        word = word_label[w]
        label_idx = []
        for ww in range(len(word)):
            label_idx.append(labels_.find(word[ww]))
        word_indices[w,:len(label_idx)] = torch.tensor(label_idx)
        word_length[w] = len(label_idx)
        
    return word_indices, word_length


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



