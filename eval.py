import os
import torch
from models import models as networks
from models.models_HiFi import Generator as model_HiFi
from modules import DTW_align, GreedyCTCDecoder, AttrDict, RMSELoss
from modules import mel2wav_vocoder, perform_STT
from utils import data_denorm, word_index
import torch.nn as nn
import torch.nn.functional as F
from NeuroTalkDataset import myDataset
import time
import torch.optim.lr_scheduler
import numpy as np
import torchaudio
from torchmetrics import CharErrorRate
import json
import argparse
from train import train as eval
import wavio
import sys


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
        
        mel_out = DTW_align(output,target)
    
        target = data_denorm(target, data_info[0], data_info[1])
        mel_out = data_denorm(mel_out, data_info[0], data_info[1])
        
        gt_label=[]
        for k in range(len(target)):
            gt_label.append(args.word_label[labels[k].item()])
            
        wav_target = mel2wav_vocoder(target, vocoder, 1)
        wav_recon = mel2wav_vocoder(mel_out, vocoder, 1)
        
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
            
def main(args):
    
    device = torch.device(f'cuda:{args.gpuNum[0]}' if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device) # change allocation of current GPU
    print ('Current cuda device ', torch.cuda.current_device()) # check
    print(torch.cuda.device_count())
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # define generator
    config_file = os.path.join(args.model_config, 'config_g.json')
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h_g = AttrDict(json_config)
    model_g = networks.Generator(h_g).cuda()
    
    args.sample_rate_mel = args.sampling_rate
    
    # define discriminator
    config_file = os.path.join(args.model_config, 'config_d.json')
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h_d = AttrDict(json_config)
    model_d = networks.Discriminator(h_d).cuda()
    
    # vocoder HiFiGAN
    # LJ_FT_T2_V3/generator_v3,   
    config_file = os.path.join(os.path.split(args.vocoder_pre)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    
    vocoder = model_HiFi(h).cuda()
    state_dict_g = torch.load(args.vocoder_pre) #, map_location=args.device)
    vocoder.load_state_dict(state_dict_g['generator'])
    
    # STT Wav2Vec
    bundle = torchaudio.pipelines.HUBERT_ASR_LARGE
    model_STT = bundle.get_model().cuda()
    args.sample_rate_STT = bundle.sample_rate
    decoder_STT = GreedyCTCDecoder(labels=bundle.get_labels())
    args.word_index, args.word_length = word_index(args.word_label, bundle)
    
    # Parallel setting
    model_g = nn.DataParallel(model_g, device_ids=args.gpuNum)
    model_d = nn.DataParallel(model_d, device_ids=args.gpuNum)
    vocoder = nn.DataParallel(vocoder, device_ids=args.gpuNum)
    model_STT = nn.DataParallel(model_STT, device_ids=args.gpuNum)

    # loss function
    criterion_recon = RMSELoss().cuda()
    criterion_adv = nn.BCELoss().cuda()
    criterion_ctc = nn.CTCLoss().cuda()
    criterion_cl = nn.CrossEntropyLoss().cuda()
    CER = CharErrorRate().cuda()

    # Directory
    saveDir = os.path.join(args.logDir, args.sub, args.task)
    # create the directory if not exist
    if not os.path.exists(saveDir):
        raise NameError('Please check the directory')

    args.savevoice = saveDir + '/savevoice'
    if not os.path.exists(args.savevoice):
        os.mkdir(args.savevoice)

    loc_g = os.path.join(saveDir, 'savemodel', 'BEST_checkpoint_g.pt')
    loc_d = os.path.join(saveDir, 'savemodel', 'BEST_checkpoint_d.pt')

    if os.path.isfile(loc_g):
        print("=> loading checkpoint '{}'".format(loc_g))
        checkpoint_g = torch.load(loc_g, map_location='cpu')
        model_g.load_state_dict(checkpoint_g['state_dict'])
        epoch = checkpoint_g['epoch']
    else:
        print("=> no checkpoint found at '{}'".format(loc_g))
        raise NameError('Can not find the trained model')
        
    if os.path.isfile(loc_d):   
        print("=> loading checkpoint '{}'".format(loc_d))
        checkpoint_d = torch.load(loc_d, map_location='cpu')
        model_d.load_state_dict(checkpoint_d['state_dict'])
    else:
        print("=> no checkpoint found at '{}'".format(loc_d))
        raise NameError('Can not find the trained model')


    # Data loader define
    testset = myDataset(mode=1, data=args.dataLoc+'/'+args.sub, task=args.task, recon=args.recon)  # file='./EEG_EC_Data_csv/train.txt'
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=4*len(args.gpuNum), pin_memory=True)

    epoch = 0

    start_time = time.time()

    print('Processing Evaluation ...')
    Te_losses = eval(args, test_loader, 
                     (model_g, model_d, vocoder, model_STT, decoder_STT), 
                     (criterion_recon, criterion_ctc, criterion_adv, criterion_cl, CER), 
                     ([],[]), 
                     epoch,
                     False,
                     True)
    
    save_test_all(args, test_loader, (model_g, model_d, vocoder, model_STT, decoder_STT), Te_losses)

    time_taken = time.time() - start_time
    print("Time: %.2f\n"%time_taken)
    

if __name__ == '__main__':

    dataDir = './dataset'
    logDir = './TrainResult'
    
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--vocoder_pre', type=str, default='./pretrained_model/UNIVERSAL_V1/g_02500000', help='pretrained vocoder file path')
    parser.add_argument('--model_config', type=str, default='./models', help='config for G & D folder path')
    parser.add_argument('--dataLoc', type=str, default=dataDir)
    parser.add_argument('--config', type=str, default='./config.json')
    parser.add_argument('--logDir', type=str, default=logDir)
    parser.add_argument('--gpuNum', type=list, default=[0])
    parser.add_argument('--batch_size', type=int, default=26)
    parser.add_argument('--sub', type=str, default='sub1')
    parser.add_argument('--task', type=str, default='SpokenEEG')
    parser.add_argument('--recon', type=str, default='Y_mel')
    parser.add_argument('--unseen', type=str, default='stop')
    
    args = parser.parse_args()
    
    with open(args.config) as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=t_args)

    
    main(args)        
    
    
    
