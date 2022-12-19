import csv
import os
import numpy as np
import torch
from torch.utils.data import Dataset

epsilon = np.finfo(float).eps

class myDataset(Dataset):
    def __init__(self, mode, data="./", task = "SpokenEEG_vec", recon="VoiceMel"):
        self.sample_rate = 8000
        self.lenth = 5 #780 # the number data
        self.lenthtest = 5 #260
        self.lenthval = 5 #260
        self.n_classes = 13
        self.mode = mode
        self.iter = iter
        self.savedata = data
        self.task = task
        self.recon = recon
        self.max_audio = 32768.0

    def __len__(self):
        if self.mode == 2:
            return self.lenthval
        elif self.mode == 1:
            return self.lenthtest
        else:
            return self.lenth

    def __getitem__(self, idx):
        '''
        :param idx:
        :return:
        '''

        if self.mode == 2:
            forder_name = self.savedata + '/val/'
        elif self.mode == 1:
            forder_name = self.savedata + '/test/'
        else:
            forder_name = self.savedata + '/train/'
            
        # tasks
        allFileList = os.listdir(forder_name + self.task + "/")
        allFileList.sort()
        file_name = forder_name + self.task + '/' + allFileList[idx]
        
        # if self.task.find('vec') != -1: # embedding vector
        #     input, avg_input, std_input = self.read_vector_data(file_name) 
        if self.task.find('mel') != -1:
            input, avg_input, std_input = self.read_data(file_name)
        elif self.task.find('Voice') != -1: # voice
            input, avg_input, std_input = self.read_voice_data(file_name)
        else: # EEG
            input, avg_input, std_input = self.read_data(file_name) 
            
            
        # recon target
        allFileList = os.listdir(forder_name + self.recon + "/")
        allFileList.sort()
        file_name = forder_name + self.recon + '/' + allFileList[idx]
        
        # if self.recon.find('vec') != -1: # embedding vector
        #     target, avg_target, std_target = self.read_vector_data(file_name) 
        if self.recon.find('mel') != -1:
            target, avg_target, std_target = self.read_data(file_name)
        elif self.recon.find('Voice') != -1: # voice
            target, avg_target, std_target = self.read_voice_data(file_name)
        else: # EEG
            target, avg_target, std_target = self.read_data(file_name) 
        
        # voice
        allFileList = os.listdir(forder_name + "Voice/")
        allFileList.sort()
        file_name = forder_name + "Voice/"+ allFileList[idx]
        voice, _, _ = self.read_voice_data(file_name)
        # voice=[]
        # target label
        allFileList = os.listdir(forder_name + "Y/")
        allFileList.sort()
        file_name = forder_name + 'Y/' + allFileList[idx]
        
        target_cl,_,_ = self.read_raw_data(file_name) 
        target_cl = np.squeeze(target_cl)


        # to tensor
        input = torch.tensor(input, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)
        

        return input, target, target_cl, voice, (avg_target, std_target, avg_input, std_input)

   
    def read_vector_data(self, file_name,n_classes):
        with open(file_name, 'r', newline='') as f:
            lines = csv.reader(f)
            data = []
            for line in lines:
                data.append(line)

        data = np.array(data).astype(np.float32)
        (r,c) = data.shape
        data = np.reshape(data,(n_classes,r//n_classes,c))
        
        max_ = np.max(data).astype(np.float32)
        min_ = np.min(data).astype(np.float32)
        avg = (max_ + min_) / 2
        std = (max_ - min_) / 2
        
        data   = np.array((data - avg) / std).astype(np.float32)

        return data, avg, std
    
    
    def read_voice_data(self, file_name):
        with open(file_name, 'r', newline='') as f:
            lines = csv.reader(f)
            data = []
            for line in lines:
                data.append(line)
        data = np.array(data).astype(np.float32)
        
        data = np.array(data / self.max_audio).astype(np.float32)
        avg = np.array([0]).astype(np.float32)

        return data, avg, self.max_audio


    def read_data(self, file_name):
        with open(file_name, 'r', newline='') as f:
            lines = csv.reader(f)
            data = []
            for line in lines:
                data.append(line)

        data = np.array(data).astype(np.float32)
        
        max_ = np.max(data).astype(np.float32)
        min_ = np.min(data).astype(np.float32)
        avg = (max_ + min_) / 2
        std = (max_ - min_) / 2
        
        data   = np.array((data - avg) / std).astype(np.float32)

            
        return data, avg, std


    def read_raw_data(self, file_name):
        with open(file_name, 'r', newline='') as f:
            lines = csv.reader(f)
            data = []
            for line in lines:
                data.append(line)

        data = np.array(data).astype(np.float32)
        avg = np.array([0]).astype(np.float32)
        std = np.array([1]).astype(np.float32)

            
        return data, avg, std


