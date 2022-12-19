
import numpy as np
import matplotlib.pyplot as plt
import os
import torch


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
    corr_num=0
    for j in range(len(wave)):
        transcript = decoder_STT(emission_recon[j])    
        transcripts.append(transcript)
        
        if transcript == gt_label[j]:
            corr_num = corr_num + 1

    acc_word = corr_num / len(wave)
        
    return transcripts, emission_recon, acc_word

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



