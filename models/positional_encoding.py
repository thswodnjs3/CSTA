import math
import torch
import torch.nn as nn

class FixedPositionalEncoding(nn.Module):
    def __init__(self, Positional_encoding_shape, dim=1024, max_len=5000, freq=10000.0):
        super(FixedPositionalEncoding, self).__init__()
        if Positional_encoding_shape=='TD':
            position = torch.arange(max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(freq) / dim))

            pe = torch.zeros(max_len, dim)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
        elif Positional_encoding_shape=='T':
            position = torch.arange(max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, 1, 2) * (-math.log(freq) / 1))

            pe = torch.zeros(max_len,1)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.repeat_interleave(dim,dim=1)
        elif Positional_encoding_shape is None:
            pass
        else:
            raise

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.shape[0]]

class RelativePositionalEncoding(nn.Module):
    def __init__(self, Positional_encoding_shape, dim=1024, max_len=5000, freq=10000.0):
        super(RelativePositionalEncoding, self).__init__()
        self.Positional_encoding_shape = Positional_encoding_shape
        self.dim = dim
        self.max_len = max_len
        self.freq = freq

    def forward(self, x):
        T = x.shape[0]
        min_rpos = -(T - 1)
        i = torch.tensor([k for k in range(T)])
        i = i.reshape(i.shape[0], 1)
        if self.Positional_encoding_shape=='TD':
            d = T + self.dim
            j = torch.tensor([k for k in range(self.dim)])

            i = i.repeat_interleave(j.shape[0], dim=1)
            j = j.repeat(i.shape[0], 1)

            r_pos = j - i - min_rpos

            pe = torch.zeros(T, self.dim)
            idx = torch.tensor([k for k in range(T//2)],dtype=torch.int64)

            pe[2*idx, :] = torch.sin(r_pos[2*idx, :] / self.freq ** ((i[2*idx, :] + j[2*idx, :]) / d))
            pe[2*idx+1, :] = torch.cos(r_pos[2*idx+1, :] / self.freq ** ((i[2*idx+1, :] + j[2*idx+1, :]) / d))
        elif self.Positional_encoding_shape=='T':
            d = T + 1
            j = torch.tensor([k for k in range(1)])

            i = i.repeat_interleave(j.shape[0], dim=1)
            j = j.repeat(i.shape[0], 1)

            r_pos = j - i - min_rpos

            pe = torch.zeros(T, 1)
            idx = torch.tensor([k for k in range(T//2)],dtype=torch.int64)

            pe[2*idx, :] = torch.sin(r_pos[2*idx, :] / self.freq ** ((i[2*idx, :] + j[2*idx, :]) / d))
            pe[2*idx+1, :] = torch.cos(r_pos[2*idx+1, :] / self.freq ** ((i[2*idx+1, :] + j[2*idx+1, :]) / d))
            pe = pe.repeat_interleave(self.dim,dim=1)
        elif self.Positional_encoding_shape is None:
            pass
        else:
            raise
        return x + pe[:x.shape[0]].to(x.device)

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, Positional_encoding_shape, dim=1024, max_len=5000):
        super(LearnablePositionalEncoding, self).__init__()
        if Positional_encoding_shape=='TD':
            self.pe = nn.Parameter(torch.randn((max_len,dim)))
        elif Positional_encoding_shape=='T':
            self.pe = nn.Parameter(torch.randn((max_len,1)))
        elif Positional_encoding_shape is None:
            pass
        else:
            raise

    def forward(self, x):
        return x + self.pe[:x.shape[0]]
    
class ConditionalPositionalEncoding(nn.Module):
    def __init__(self, Positional_encoding_shape, Positional_encoding_way, dim=1024, kernel_size=3, stride=1, padding=1):
        super(ConditionalPositionalEncoding, self).__init__()
        self.Positional_encoding_way = Positional_encoding_way
        if Positional_encoding_shape=='TD':
            self.pe = nn.Conv1d(in_channels=dim,out_channels=dim,kernel_size=kernel_size,stride=stride,padding=padding)
        elif Positional_encoding_shape=='T':
            self.pe = nn.Conv1d(in_channels=dim,out_channels=dim,kernel_size=kernel_size,stride=stride,padding=padding,groups=dim)
        else:
            raise

    def forward(self, x):
        if self.Positional_encoding_way=='Transformer':
            return x + self.pe(x[0].permute(0,2,1)).permute(0,2,1).unsqueeze(0)
        elif self.Positional_encoding_way=='PGL_SUM':
            return x + self.pe(x.unsqueeze(0).permute(0,2,1)).permute(0,2,1).squeeze(0)
        elif self.Positional_encoding_way is None:
            pass
        else:
            raise
