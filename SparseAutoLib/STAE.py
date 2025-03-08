import torch
import torch.nn as nn
from utils import *
from SparseAutoLib.modules import *

class STAE(nn.Module):
    def __init__(self, enc_in):
        super(STAE, self).__init__()
        self.channel = enc_in  #  input channels
        num_inputs = 12 
        num_channels = [32, 64, 128]  

        # 1D Temporal Encoder for ECG signals
        self.time_encoder = Encoder1D(num_inputs, num_channels=[32, 64, 128], kernel_size=2)
        
        # 1D Temporal Decoder for reconstruction
        self.time_decoder = Decoder1D(136)

        # 2D Spectrogram Temporal Encoder 
        self.spec_encoder = Encoder2D(num_inputs, num_channels=[16, 32, 64, 128], kernel_size=(2,2))

        self.conv_spec1 = nn.Conv1d(50 * 51, 50, 3, 1, 1, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(202, 136),  # Linear transformation
            nn.LayerNorm(136),  # Normalization
            nn.ReLU()  # Activation
        )
        # Strided Attention Block
        self.attn1 = StridedBlockAttention(5, 50)
        self.drop = nn.Dropout(0.2)
        self.layer_norm1 = LayerNorm(50)

    def attention_func(self, x, attn, norm):
        """Apply attention mechanism with residual connection and normalization."""
        attn_latent = attn(x, x, x)
        attn_latent = norm(x + self.drop(attn_latent))  # Residual connection with dropout
        return attn_latent

    def forward(self, time_ecg, spectrogram_ecg):
        """Forward pass of the model."""
        
        # Encode time-domain ECG features
        time_features = self.time_encoder(time_ecg.transpose(-1,1))  

        # Encode spectrogram ECG features
        spectrogram_features = self.spec_encoder(spectrogram_ecg.permute(0,3,1,2))  
        n, c, h, w = spectrogram_features.shape
        spectrogram_features = self.conv_spec1(spectrogram_features.contiguous().view(n, c*h, w))

        # Concatenate encoded features from both encoders
        latent_combine = torch.cat([time_features, spectrogram_features], dim=-1)
        latent_combine = latent_combine.transpose(-1, 1)  # Rearrange dimensions for attention

        # Apply attention twice
        attn_latent = self.attention_func(latent_combine, self.attn1, self.layer_norm1)
        attn_latent = self.attention_func(attn_latent, self.attn1, self.layer_norm1)

        latent_combine = attn_latent.transpose(-1, 1)
        latent_combine = self.mlp(latent_combine)
        latent_combine = latent_combine.permute(0, 2, 1)  

        # Decode time-domain features
        output = self.time_decoder(latent_combine)
        output = output.transpose(-1, 1)

        # Return reconstructed output and  uncertainty
        return (output[:, :, 0:self.channel], output[:, :, self.channel:self.channel+1])
