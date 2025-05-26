import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from lib.modules import *

class STAE(nn.Module):

    def __init__(self, enc_in):
        super(STAE, self).__init__()

        self.channel = enc_in

        # Time series 
        num_inputs = 12
        num_channels = [64, 128, 256]

        self.time_encoder = Encoder1D(num_inputs,num_channels =[32, 64, 128], kernel_size=2)
        
        self.time_decoder = Decoder1D(136)

        # Spectrogram 
        self.spec_encoder = Encoder2D(num_inputs, num_channels=[16, 32, 64, 128] , kernel_size=(2,2))

        self.conv_spec1 = nn.Conv1d(50*51, 50, 3, 1, 1, bias=False)

        self.mlp = nn.Sequential(
            nn.Linear(202, 136),
            nn.LayerNorm(136),
            nn.ReLU()
        )

        self.attn1 = StridedBlockAttention(5,50)
        self.drop = nn.Dropout(0.2)
        self.layer_norm1 = LayerNorm(50)

    def attention_func(self,x, attn, norm):
        attn_latent = attn(x, x, x)
        attn_latent = norm(x + self.drop(attn_latent))
        return attn_latent

    def forward(self, time_ecg, spectrogram_ecg):

        #Time ECG encoder

        time_features = self.time_encoder(time_ecg.transpose(-1,1)) #(32, 50, 136)

        #Spectrogram ECG encoder

        spectrogram_features = self.spec_encoder(spectrogram_ecg.permute(0,3,1,2)) #(32, 50, 63, 66)


        n, c, h, w = spectrogram_features.shape
        spectrogram_features = self.conv_spec1(spectrogram_features.contiguous().view(n, c*h, w)) #(32, 50, 66)
        

        latent_combine = torch.cat([time_features, spectrogram_features], dim=-1)

        #Sparse-attention
        latent_combine = latent_combine.transpose(-1, 1)
        attn_latent = self.attention_func(latent_combine, self.attn1, self.layer_norm1)
        attn_latent = self.attention_func(attn_latent, self.attn1, self.layer_norm1)

        latent_combine = attn_latent.transpose(-1, 1)

        latent_combine = self.mlp(latent_combine)

        latent_combine = latent_combine.permute(0, 2, 1)  # [32, 136, 50]
        output = self.time_decoder(latent_combine)

        output = output.transpose(-1, 1)
        
        return  (output[:,:,0:self.channel],output[:,:,self.channel:self.channel+1])
