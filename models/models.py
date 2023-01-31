import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm
from utils import init_weights, get_padding
import math

LRELU_SLOPE = 0.1


class ResBlock(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1,3,5)):
        super(ResBlock, self).__init__()
        self.h = h
        self.convs1 = nn.ModuleList([
            weight_norm(
                Conv1d(channels, channels,
                       kernel_size, 1, 
                       dilation=dilation[0],                               
                       padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(
                Conv1d(channels, channels,                                
                       kernel_size, 1,                                
                       dilation=dilation[1],                               
                       padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(
                Conv1d(channels, channels,                                
                       kernel_size, 1,                                
                       dilation=dilation[2],                               
                       padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(
                Conv1d(channels, channels,                                
                       kernel_size, 1, 
                       dilation=1,
                       padding=get_padding(kernel_size, 1))),
            weight_norm(
                Conv1d(channels, channels, 
                       kernel_size, 1, 
                       dilation=1,
                       padding=get_padding(kernel_size, 1))),
            weight_norm(
                Conv1d(channels, channels, 
                       kernel_size, 1, 
                       dilation=1,
                       padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)
            
            
class Generator(torch.nn.Module):
    def __init__(self, h):
        super(Generator, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.i_mid = 0
        self.i_mid_gru = 1
        
        # model define
        self.conv_pre = weight_norm(
            Conv1d(h.in_ch, 
                   h.ch_init_upsample//2,
                   3, 1, 
                   padding=get_padding(3,1)))
        
        
        self.GRU = nn.GRU(h.ch_init_upsample//2, 
                          h.ch_init_upsample//4, 
                          num_layers=1, 
                          batch_first=True, 
                          bidirectional=True)
        
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, 
                                       h.upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(h.ch_init_upsample//(2**i), 
                                h.ch_init_upsample//(2**(i+1)),
                                k, u, padding=(k-u)//2)))
            
        self.conv_mid1 = weight_norm(
            Conv1d(h.ch_init_upsample//(2**self.i_mid), 
                   h.ch_init_upsample//(2**self.i_mid), 
                   3, 1, 
                   padding=0))
        
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.ch_init_upsample//(2**(i+1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, 
                                           h.resblock_dilation_sizes)):
                self.resblocks.append(ResBlock(h, ch, k, d))

        self.conv_post = weight_norm(
            Conv1d(ch, 
                   h.out_ch, 
                   9, 1, 
                   padding=get_padding(9,1)))
        
        self.conv_pre.apply(init_weights)
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.conv_mid1.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        x_temp = x
        x = x.transpose(1, 2)
        self.GRU.flatten_parameters()
        x, _ = self.GRU(x)
        x = x.transpose(1, 2)
        x = torch.cat([x, x_temp], dim=1)

        for i in range(self.num_upsamples):
            # to match the output size
            if i == self.i_mid:
                x = self.conv_mid1(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
        remove_weight_norm(self.conv_mid1)


class Discriminator(torch.nn.Module):
    def __init__(self, h):
        super(Discriminator, self).__init__()
        self.h = h
        self.ch_init_downsample = h.ch_init_downsample
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_downsamples = len(h.downsample_rates)
        self.n_classes = h.n_classes
        self.input_size = h.input_size
        self.m = 1
        
        for j in range(len(h.downsample_rates)):
            self.m = self.m * h.downsample_rates[j]
        
        # model define
        self.conv_pre = weight_norm(
            Conv1d(h.in_ch, 
                   h.ch_init_downsample,
                   3, 1, 
                   padding=get_padding(3,1)))
        
        self.downs = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.downsample_rates, 
                                       h.downsample_kernel_sizes)):
            self.downs.append(weight_norm(
                Conv1d(h.ch_init_downsample*(2**i), 
                       h.ch_init_downsample*(2**(i+1)),
                       k, u, padding=math.ceil((k-u)/2))))
            
        self.resblocks = nn.ModuleList()
        for i in range(len(self.downs)):
            ch = h.ch_init_downsample*(2**(i+1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, 
                                           h.resblock_dilation_sizes)):
                self.resblocks.append(ResBlock(h, ch, k, d))
        
        self.GRU = nn.GRU(ch, ch//2,
                          num_layers=1, 
                          batch_first=True, 
                          bidirectional=True)
        
        self.conv_post = weight_norm(Conv1d(ch, ch, 9, 1, padding=get_padding(9,1)))
        
        # FC Layer 
        self.adv_classifier = nn.Sequential(nn.Linear(
            h.ch_init_downsample*2*8*(self.input_size//self.m), 1),
            nn.Sigmoid())
        self.aux_classifier = nn.Sequential(nn.Linear(
            h.ch_init_downsample*2*8*(self.input_size//self.m), h.n_classes),
            nn.Softmax(dim=1))
        
        self.conv_pre.apply(init_weights)
        self.downs.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)

        for i in range(self.num_downsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.downs[i](x)

            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x_temp = x
        x = x.transpose(1, 2)
        self.GRU.flatten_parameters()
        x, _ = self.GRU(x)
        x = x.transpose(1, 2)
        x = torch.cat([x, x_temp], dim=1)

        # FC Layer
        x = x.view(-1,
                   self.ch_init_downsample
                   *2*8*(self.input_size//self.m))
        validity = self.adv_classifier(x)
        label = self.aux_classifier(x)
        
        return validity, label

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.downs:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
            