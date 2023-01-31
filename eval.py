import os
import torch
from models import models as networks
from models.models_HiFi import Generator as model_HiFi
from modules import GreedyCTCDecoder, AttrDict, RMSELoss, save_test_all
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


    loc_g = os.path.join(args.trained_model, args.task, 'checkpoint_g.pth')
    loc_d = os.path.join(args.trained_model, args.task, 'checkpoint_d.pth')
    
    if os.path.isfile(loc_g):
        print("=> loading checkpoint '{}'".format(loc_g))
        checkpoint_g = torch.load(loc_g, map_location='cpu')
        model_g.load_state_dict(checkpoint_g['model_state_dict'])
    else:
        print("=> no checkpoint found at '{}'".format(loc_g))
        
    if os.path.isfile(loc_d):   
        print("=> loading checkpoint '{}'".format(loc_d))
        checkpoint_d = torch.load(loc_d, map_location='cpu')
        model_d.load_state_dict(checkpoint_d['model_state_dict'])
    else:
        print("=> no checkpoint found at '{}'".format(loc_d))
        

    criterion_recon = RMSELoss().cuda()
    criterion_adv = nn.BCELoss().cuda()
    criterion_ctc = nn.CTCLoss().cuda()
    criterion_cl = nn.CrossEntropyLoss().cuda()
    CER = CharErrorRate().cuda()
    
    saveDir = args.save + '_' + args.sub + '_' + args.task
    # create the directory if not exist
    if not os.path.exists(saveDir):
        os.mkdir(saveDir)
    
    args.savevoice = saveDir + '/savevoice'
    if not os.path.exists(args.savevoice):
        os.mkdir(args.savevoice)
        

    # Data loader define
    testset = myDataset(mode=1, data=args.dataLoc+'/'+args.sub, task=args.task, recon=args.recon)  # file='./EEG_EC_Data_csv/train.txt'
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=4*len(args.gpuNum), pin_memory=True)
    
    max_epochs = 1
    epoch = 0
        
    start_time = time.time()
    
    print("Epoch : %d/%d" %(epoch, max_epochs) )
    
    Te_losses = eval(args, test_loader, 
                     (model_g, model_d, vocoder, model_STT, decoder_STT), 
                     (criterion_recon, criterion_ctc, criterion_adv, criterion_cl, CER), 
                     ([],[]), 
                     epoch,
                     False)
    
    save_test_all(args, test_loader, (model_g, model_d, vocoder, model_STT, decoder_STT), Te_losses)

    time_taken = time.time() - start_time
    print("Time: %.2f\n"%time_taken)
    

if __name__ == '__main__':
    
    fileDir = './sample_data'
    saveDir = './TrainResult'
    
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--vocoder_pre', type=str, default='./pretrained_model/UNIVERSAL_V1/g_02500000', help='pretrained vocoder file path')
    parser.add_argument('--trained_model', type=str, default='./pretrained_model', help='config for G & D folder path')
    parser.add_argument('--model_config', type=str, default='./models', help='config for G & D folder path')
    parser.add_argument('--dataLoc', type=str, default=fileDir)
    parser.add_argument('--config', type=str, default='./config.json')
    parser.add_argument('--save', type=str, default=saveDir)
    parser.add_argument('--gpuNum', type=list, default=[0,1,2])
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
    
    
    
