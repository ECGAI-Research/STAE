import torch
import torch.nn.functional as F
import torch.nn as nn
import math

class StridedBlockAttention(nn.Module):
    def __init__(self, h, d_model , block_size=400, stride=50,  dropout=0.1):
        super().__init__()
        assert d_model % h == 0, "d_model must be divisible by the number of heads"
        self.block_size = block_size
        self.stride = stride
        self.h = h
        self.d_k = d_model // h
        self.dropout = nn.Dropout(dropout)
        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)

    def create_strided_block_mask(self, seq_len, device):
        mask = torch.zeros(seq_len, seq_len, device=device)
        num_blocks = (seq_len + self.block_size - 1) // self.block_size
        for block_idx in range(num_blocks):
            start = block_idx * self.block_size
            end = min(start + self.block_size, seq_len)
            for i in range(start, end, self.stride):
                mask[i, start:end:self.stride] = 1  
        return mask

    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, _ = query.size()
        device = query.device
        
        # Apply linear transformations and reshape for multi-head attention
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]
        
        # Compute scaled dot-product attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply strided attention mask
        strided_mask = self.create_strided_block_mask(seq_len, device)
        scores = scores.masked_fill(strided_mask == 0, -1e9)

        # Compute global scores and combine with local scores
        global_scores = torch.mean(scores, dim=-2, keepdim=True)  
        combined_scores = scores + global_scores  
    
        attn = F.softmax(combined_scores, dim=-1)
        attn = self.dropout(attn)
        
        # Compute attention output
        x = torch.matmul(attn, value)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.output_linear(x)

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.weight * (x - mean) / (std + self.eps) + self.bias

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()
# Temporal block for Encoder1D 
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        # Define a temporal convolutional block with residual connections
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.net1 = nn.Sequential(self.conv1, self.chomp1, self.bn1, self.relu1, self.dropout1)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net2 = nn.Sequential(self.conv2, self.chomp2, self.bn2, self.relu2, self.dropout2)

        self.conv3 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp3 = Chomp1d(padding)
        self.bn3 = nn.BatchNorm1d(n_outputs)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)
        self.net3 = nn.Sequential(self.conv3, self.chomp3, self.bn3, self.relu3, self.dropout3)

        # Apply downsampling if input and output dimensions do not match
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net1(x)
        out = self.net2(out)
        out = self.net3(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)  # Residual connection with ReLU activation

class Encoder1D(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(Encoder1D, self).__init__()
        self.num_levels = len(num_channels)
        self.network = nn.ModuleList()
        for i in range(self.num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            self.network.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                              padding=(kernel_size-1) * dilation_size, dropout=dropout))
        self.fc = nn.Linear(num_channels[-1], 136)  # Output layer with 136 features

    def forward(self, x):
        out = x
        for i in range(self.num_levels):
            out = self.network[i](out)
        out = F.adaptive_avg_pool1d(out, 50)  
        out = out.permute(0, 2, 1)  
        out = self.fc(out)
        return out
    
class Chomp1dD(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1dD, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

# Temporal block for Decoder1D 
class TemporalBlockD(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding):
        super(TemporalBlockD, self).__init__()

        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1dD(padding)
        self.relu1 = nn.ReLU()
        self.net1 = nn.Sequential(self.conv1, self.chomp1, self.relu1)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1dD(padding)
        self.relu2 = nn.ReLU()
        self.net2 = nn.Sequential(self.conv2, self.chomp2, self.relu2)

        self.conv3 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp3 = Chomp1dD(padding)
        self.relu3 = nn.ReLU()
        self.net3 = nn.Sequential(self.conv3, self.chomp3, self.relu3)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net1(x)
        out = self.net2(out)
        out = self.net3(out)
        res = x if self.downsample is None else self.downsample(x)  # Residual connection
        return self.relu(out + res)

class Decoder1D(nn.Module):
    def __init__(self, num_inputs, num_channels=[256, 128, 64, 32, 16], kernel_size=2):
        super(Decoder1D, self).__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(TemporalBlockD(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=(kernel_size-1) * dilation_size))

        self.network = nn.Sequential(*layers)
        self.conv_out = nn.Conv1d(num_channels[-1], 13, 1)  # Final output layer with 13 channels

    def forward(self, x):
        out = self.network(x)
        out = F.interpolate(out, size=4800, mode='linear', align_corners=False)  
        return self.conv_out(out)

class Chomp2d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp2d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size[0], :-self.chomp_size[1]].contiguous()

# Temporal block for Decoder2D
class TemporalBlock2D(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock2D, self).__init__()

        self.conv1 = nn.Conv2d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp2d(padding)
        self.bn1 = nn.BatchNorm2d(n_outputs)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout2d(dropout)
        self.net1 = nn.Sequential(self.conv1, self.chomp1, self.bn1, self.relu1, self.dropout1)

        self.conv2 = nn.Conv2d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp2d(padding)
        self.bn2 = nn.BatchNorm2d(n_outputs)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout2d(dropout)
        self.net2 = nn.Sequential(self.conv2, self.chomp2, self.bn2, self.relu2, self.dropout2)

        self.conv3 = nn.Conv2d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp3 = Chomp2d(padding)
        self.bn3 = nn.BatchNorm2d(n_outputs)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout2d(dropout)
        self.net3 = nn.Sequential(self.conv3, self.chomp3, self.bn3, self.relu3, self.dropout3)

        self.downsample = nn.Conv2d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net1(x)
        out = self.net2(out)
        out = self.net3(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class Encoder2D(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=(2, 2), dropout=0.2):
        super(Encoder2D, self).__init__()
        self.network = nn.ModuleList()
        for i in range(len(num_channels)):
            dilation_size = (2 ** i, 1)  # Dilation increases in height only
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = ((kernel_size[0]-1) * dilation_size[0], (kernel_size[1]-1) * dilation_size[1])
            self.network.append(TemporalBlock2D(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                                padding=padding, dropout=dropout))

        self.fc = nn.Linear(sum(num_channels), 66)  # Fully connected output layer

    def forward(self, x):
        skip_connections = []
        for layer in self.network:
            x = layer(x)
            skip_connections.append(x)

        out = torch.cat(skip_connections, dim=1)  # Concatenate intermediate outputs
        out = F.adaptive_avg_pool2d(out, (50, 51))  
        out = out.permute(0, 2, 3, 1) 
        return self.fc(out)
