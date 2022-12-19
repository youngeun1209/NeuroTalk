import os
import torch
from models import models as networks
from models.models_HiFi import Generator as model_HiFi
from modules import DTW_align, GreedyCTCDecoder, AttrDict, RMSELoss
from utils import data_denorm, mel2wav_vocoder, perform_STT
import torch.nn as nn
import torch.nn.functional as F
from NeuroTalkDataset import myDataset
import time
import torch.optim.lr_scheduler
import numpy as np
import torchaudio
from torchmetrics import CharErrorRate, WordErrorRate
import json
import argparse
import wavio
    
def eval(args, test_loader, models, criterions, epoch):
    '''
    :param args: general arguments
    :param train_loader: loaded for training/validation/test dataset
    :param model: model
    :param criterion: loss function
    :param optimizer: optimization algo, such as ADAM or SGD
    :param epoch: epoch number
    :return: average epoch loss, overall pixel-wise accuracy, per class accuracy, per class iu, and mIOU
    '''
    
    # switch to train mode
    if type(models) != tuple:
        print("Two models should be inputed (generator and discriminator)")
        
    (model_g, model_d, vocoder, model_STT, decoder_STT) = models
    (criterion_recon, criterion_adv, CER, WER) =  criterions
    
    model_g.eval()
    model_d.eval()
    vocoder.eval()
    model_STT.eval()
    
    epoch_loss_g = []
    epoch_loss_g_recon = []
    epoch_loss_g_valid = []
    epoch_loss_d = []
    epoch_loss_d_valid = []
    
    epoch_acc_g_valid=[]
    epoch_cer_gt = []
    epoch_cer_recon = []
    epoch_acc_d_real = []
    epoch_acc_d_fake = []

    total_batches = len(test_loader)
    
    for i, (input, target, target_cl, voice, data_info) in enumerate(test_loader):    
        
        print("\rBatch [%5d / %5d]"%(i,total_batches), sep=' ', end='', flush=True)
        
        # Adversarial ground truths 1:real, 0: fake
        valid = torch.ones((len(input), 1), dtype=torch.float32).cuda()
        fake = torch.zeros((len(input), 1), dtype=torch.float32).cuda()
        
        input = input.cuda()
        target = target.cuda()
        target_cl = target_cl.cuda()
        voice = torch.squeeze(voice,dim=-1).cuda()
        labels = torch.argmax(target_cl,dim=1) 

        ###############################
        # Train Generator
        ###############################

        with torch.no_grad():
            output = model_g(input)
            g_valid = model_d(output)
        
        # when not overt, DTW is needed
        out_DTW = output.clone()
        if args.task[0] == 'I': 
            out_DTW = DTW_align(out_DTW, target)
        
        # generator loss
        loss1 = criterion_recon(out_DTW, target)
        
        # GAN loss
        loss_valid = criterion_adv(g_valid,valid)
        
        
        ###############################
        # Loss from Vocoder - STT
        ###############################
        # out_DTW
        target_denorm = data_denorm(target, data_info[0], data_info[1])
        output_denorm = data_denorm(out_DTW, data_info[0], data_info[1])
        
        gt_label=[]
        for j in range(len(target)):
            gt_label.append(args.word_label[labels[j].item()])
            
        # target
        ##### HiFi-GAN
        with torch.no_grad():
            wav_target = vocoder(target_denorm)
            wav_target = torch.reshape(wav_target, (len(wav_target),wav_target.shape[-1]))
        
        #### resampling
        wav_target = torchaudio.functional.resample(wav_target, args.sample_rate_mel, args.sample_rate_STT)   
        if wav_target.shape[1] !=  voice.shape[1]:
            p = voice.shape[1] - wav_target.shape[1]
            p_s = p//2
            p_e = p-p_s
            wav_target = F.pad(wav_target, (p_s,p_e))
        
        # recon
        ##### HiFi-GAN
        with torch.no_grad():
            wav_recon = vocoder(output_denorm)
            wav_recon = torch.reshape(wav_recon, (len(wav_recon),wav_recon.shape[-1]))
        
        #### resampling
        wav_recon = torchaudio.functional.resample(wav_recon, args.sample_rate_mel, args.sample_rate_STT)   
        if wav_recon.shape[1] !=  voice.shape[1]:
            p = voice.shape[1] - wav_recon.shape[1]
            p_s = p//2
            p_e = p-p_s
            wav_recon = F.pad(wav_recon, (p_s,p_e))

        ##### STT Wav2Vec 2.0
        with torch.no_grad():
            emission_gt, _ = model_STT(voice)
            emission_recon, _ = model_STT(wav_recon)

        # decoder STT
        transcript_gt = []
        transcript_recon = []

        for j in range(len(voice)):
            transcript = decoder_STT(emission_gt[j])   
            transcript_gt.append(transcript)
                
            transcript = decoder_STT(emission_recon[j])
            transcript_recon.append(transcript)
        
        cer_gt = CER(transcript_gt, gt_label)
        cer_recon = CER(transcript_recon, gt_label)

        # total generator loss
        loss_g = args.l_g[0] * loss1 + args.l_g[1] * loss_valid + args.l_g[2] * cer_recon 
        
        # accuracy
        acc_g_valid = (g_valid.round() == valid).float().mean()
        
        epoch_loss_g.append(loss_g.item())
        epoch_loss_g_recon.append(loss1.item())
        epoch_loss_g_valid.append(loss_valid.item())
        epoch_acc_g_valid.append(acc_g_valid.item())
        
        if torch.isnan(cer_gt):
            epoch_cer_gt.append(np.array([0]))
        else:
            epoch_cer_gt.append(cer_gt.item())

        if torch.isnan(cer_recon):
            epoch_cer_recon.append(np.array([0]))
        else:
            epoch_cer_recon.append(cer_recon.item())


        ###############################
        # Train Discriminator
        ###############################
        with torch.no_grad():
            real_valid = model_d(target)
            fake_valid = model_d(out_DTW.detach())
        
        loss_d_real_valid = criterion_adv(real_valid, valid)
        loss_d_fake_valid = criterion_adv(fake_valid, fake)
        
        loss_d_valid = 0.5 * (loss_d_real_valid + loss_d_fake_valid)
        loss_d = loss_d_valid
        
        # accuracy
        acc_d_valid = (real_valid.round() == valid).float().mean()
        acc_d_fake = (fake_valid.round() == fake).float().mean()
        
        epoch_loss_d.append(loss_d.item())
        epoch_loss_d_valid.append(loss_d_valid.item())
        epoch_acc_d_real.append(acc_d_valid.item())
        epoch_acc_d_fake.append(acc_d_fake.item())
        

        
    args.loss_g = sum(epoch_loss_g) / len(epoch_loss_g)
    args.loss_g_recon = sum(epoch_loss_g_recon) / len(epoch_loss_g_recon)
    args.loss_g_valid = sum(epoch_loss_g_valid) / len(epoch_loss_g_valid)
    args.acc_g_valid = sum(epoch_acc_g_valid) / len(epoch_acc_g_valid)
    args.cer_gt = sum(epoch_cer_gt) / len(epoch_cer_gt)
    args.cer_recon = sum(epoch_cer_recon) / len(epoch_cer_recon)
    
    args.loss_d = sum(epoch_loss_d) / len(epoch_loss_d)
    args.loss_d_valid = sum(epoch_loss_d_valid) / len(epoch_loss_d_valid)
    args.acc_d_real = sum(epoch_acc_d_real) / len(epoch_acc_d_real)
    args.acc_d_fake = sum(epoch_acc_d_fake) / len(epoch_acc_d_fake)

    return (args.loss_g, args.loss_g_recon, args.loss_g_valid, args.acc_g_valid, args.cer_gt, args.cer_recon, args.loss_d, args.loss_d_valid, args.acc_d_real, args.acc_d_fake)


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
        transcript_target, _, _ = perform_STT(wav_target, model_STT, decoder_STT, gt_label, 1)
        transcript_recon, _, _ = perform_STT(wav_recon, model_STT, decoder_STT, gt_label, 1)
        transcript_voice, _, _ = perform_STT(voice, model_STT, decoder_STT, gt_label, 1)
        
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
    
    seed = 42
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # define generator
    config_file = os.path.join(os.path.split(args.trained_model)[0], 'config_g.json')
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h_g = AttrDict(json_config)
    model_g = networks.Generator(h_g).cuda()
    
    args.sample_rate_mel = h_g.sampling_rate
    args.l_g = h_g.l_g
    args.word_label  = h_g.word_label
    
    # define discriminator
    config_file = os.path.join(os.path.split(args.trained_model)[0], 'config_d.json')
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
    state_dict_g = torch.load(args.vocoder_pre) 
    vocoder.load_state_dict(state_dict_g['generator'])
    
    # STT Wav2Vec
    bundle = torchaudio.pipelines.HUBERT_ASR_LARGE
    model_STT = bundle.get_model().cuda()
    args.sample_rate_STT = bundle.sample_rate
    decoder_STT = GreedyCTCDecoder(labels=bundle.get_labels())
    
    # Parallel setting
    model_g = nn.DataParallel(model_g, device_ids=args.gpuNum)
    model_d = nn.DataParallel(model_d, device_ids=args.gpuNum)
    vocoder = nn.DataParallel(vocoder, device_ids=args.gpuNum)
    model_STT = nn.DataParallel(model_STT, device_ids=args.gpuNum)
    
    loc_g = args.trained_model + 'checkpoint_g.pth.tar'
    loc_d = args.trained_model + 'checkpoint_d.pth.tar'
    
    if os.path.isfile(loc_g):
        print("=> loading checkpoint '{}'".format(loc_g))
        checkpoint_g = torch.load(loc_g, map_location='cpu')
        model_g.load_state_dict(checkpoint_g['state_dict'])
    else:
        print("=> no checkpoint found at '{}'".format(loc_g))
        
    if os.path.isfile(loc_d):   
        checkpoint_d = torch.load(loc_d, map_location='cpu')
        model_d.load_state_dict(checkpoint_d['state_dict'])
    else:
        print("=> no checkpoint found at '{}'".format(loc_d))
        
    criterion_recon = RMSELoss().cuda()
    criterion_adv = nn.BCELoss().cuda()
    CER = CharErrorRate().cuda()
    WER = WordErrorRate().cuda()
    
    saveDir = args.save + '_' + args.sub
    # create the directory if not exist
    if not os.path.exists(saveDir):
        os.mkdir(saveDir)
    
    args.savevoice = saveDir + '/savevoice'
    if not os.path.exists(args.savevoice):
        os.mkdir(args.savevoice)
    
    # Data loader define
    testset = myDataset(mode=1, data=args.dataLoc+'/'+args.sub, task=args.task, recon=args.recon)  
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=4*len(args.gpuNum), pin_memory=True)
    
    
    max_epochs = 1
    epoch = 0
    
    start_time = time.time()
    
    print("Epoch : %d/%d" %(epoch, max_epochs) )

    Ts_losses = eval(args, test_loader, 
                     (model_g, model_d, vocoder, model_STT, decoder_STT), 
                     (criterion_recon, criterion_adv, CER, WER), 
                     epoch) 

    time_taken = time.time() - start_time
    print("Time: %.2f\n"%time_taken)
    
    save_test_all(args, test_loader, (model_g, model_d, vocoder, model_STT, decoder_STT), Ts_losses)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('trained_model', help='config & checkpoint for G & D folder path')
    parser.add_argument('vocoder_pre', help='pretrained vocoder file path')
    parser.add_argument('--gpuNum', type=int, default=[0])
    parser.add_argument('--dataLoc', type=str, default='./sample_data')
    parser.add_argument('--sub', type=str, default='sub1')
    parser.add_argument('--task', type=str, default='SpokenEEG_vec')
    parser.add_argument('--recon', type=str, default='Voice_mel')
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--save', type=str, default='./TestResult')

    args = parser.parse_args()

    main(args)        
    
    
    
    















