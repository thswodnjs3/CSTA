import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Any, Callable, List, Optional, Tuple

from models.positional_encoding import FixedPositionalEncoding,LearnablePositionalEncoding,RelativePositionalEncoding,ConditionalPositionalEncoding

# Edit GoogleNet by replacing last parts with adaptive average pooling layers
class GoogleNet_Att(nn.Module):
    __constants__ = ["aux_logits", "transform_input"]

    def __init__(
        self,
        num_classes: int = 1000,
        init_weights: Optional[bool] = None
    ) -> None:
        super().__init__()
        conv_block = BasicConv2d
        inception_block = Inception

        self.conv1 = conv_block(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = conv_block(64, 64, kernel_size=1)
        self.conv3 = conv_block(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = inception_block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = inception_block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = inception_block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = inception_block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = inception_block(832, 384, 192, 384, 48, 128, 128)

        self.fc = nn.Linear(1024, num_classes)

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    torch.nn.init.trunc_normal_(m.weight, mean=0.0, std=0.01, a=-2, b=2)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def _forward(self, x: Tensor, n_frame) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)

        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)

        ##############################################################################
        # The place I edit to resize feature maps, and to handle various lengths of input videos
        ##############################################################################
        self.avgpool = nn.AdaptiveAvgPool2d((n_frame,1))
        x = self.avgpool(x)
        x = torch.squeeze(x)
        x = x.permute(1,0)
        return x
    
    def forward(self, x: Tensor):
        ##############################################################################
        # Takes the number of frames to handle various lengths of input videos
        ##############################################################################
        n_frame = x.shape[2]
        x = self._forward(x,n_frame)
        return x

class Inception(nn.Module):
    def __init__(
        self,
        in_channels: int,
        ch1x1: int,
        ch3x3red: int,
        ch3x3: int,
        ch5x5red: int,
        ch5x5: int,
        pool_proj: int,
        conv_block: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1 = conv_block(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            conv_block(in_channels, ch3x3red, kernel_size=1), conv_block(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, ch5x5red, kernel_size=1),
            conv_block(ch5x5red, ch5x5, kernel_size=3, padding=1),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            conv_block(in_channels, pool_proj, kernel_size=1),
        )

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return outputs
    
    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)
    
class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

##############################################################################
#   Define our proposed model
##############################################################################
class CSTA_GoogleNet(nn.Module):
    def __init__(self,
                 model_name,
                 Scale,
                 Softmax_axis,
                 Balance,
                 Positional_encoding,
                 Positional_encoding_shape,
                 Positional_encoding_way,
                 Dropout_on,
                 Dropout_ratio,
                 Classifier_on,
                 CLS_on,
                 CLS_mix,
                 key_value_emb,
                 Skip_connection,
                 Layernorm,
                 dim=1024):
        super().__init__()
        self.googlenet = GoogleNet_Att()

        self.model_name = model_name
        self.Scale = Scale
        self.Softmax_axis = Softmax_axis
        self.Balance = Balance

        self.Positional_encoding = Positional_encoding
        self.Positional_encoding_shape = Positional_encoding_shape
        self.Positional_encoding_way = Positional_encoding_way
        self.Dropout_on = Dropout_on
        self.Dropout_ratio = Dropout_ratio

        self.Classifier_on = Classifier_on
        self.CLS_on = CLS_on
        self.CLS_mix = CLS_mix

        self.key_value_emb = key_value_emb
        self.Skip_connection = Skip_connection
        self.Layernorm = Layernorm

        self.dim = dim

        if self.Positional_encoding is not None:
            if self.Positional_encoding=='FPE':
                self.Positional_encoding_op = FixedPositionalEncoding(
                    Positional_encoding_shape=self.Positional_encoding_shape,
                    dim=self.dim
                )
            elif self.Positional_encoding=='RPE':
                self.Positional_encoding_op = RelativePositionalEncoding(
                    Positional_encoding_shape=self.Positional_encoding_shape,
                    dim=self.dim
                )
            elif self.Positional_encoding=='LPE':
                self.Positional_encoding_op = LearnablePositionalEncoding(
                    Positional_encoding_shape=self.Positional_encoding_shape,
                    dim=self.dim
                )
            elif self.Positional_encoding=='CPE':
                self.Positional_encoding_op = ConditionalPositionalEncoding(
                    Positional_encoding_shape=self.Positional_encoding_shape,
                    Positional_encoding_way=self.Positional_encoding_way,
                    dim=self.dim
                )
            elif self.Positional_encoding is None:
                pass
            else:
                raise

        if self.Positional_encoding_way=='Transformer':
            self.Positional_encoding_embedding = nn.Linear(in_features=self.dim, out_features=self.dim)
        elif self.Positional_encoding_way=='PGL_SUM' or self.Positional_encoding_way is None:
            pass 
        else:
            raise
        
        if self.Dropout_on:
            self.dropout = nn.Dropout(p=float(self.Dropout_ratio))
        
        if self.Classifier_on:
            self.linear1 = nn.Sequential(
                nn.Linear(in_features=self.dim, out_features=self.dim),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.LayerNorm(normalized_shape=self.dim, eps=1e-6)
            )
            self.linear2 = nn.Sequential(
                nn.Linear(in_features=self.dim, out_features=1),
                nn.Sigmoid()
            )

            for name,param in self.named_parameters():
                if name in ['linear1.0.weight','linear2.0.weight']:
                    nn.init.xavier_uniform_(param, gain=np.sqrt(2.0))
                elif name in ['linear1.0.bias','linear2.0.bias']:
                    nn.init.constant_(param, 0.1)
        else:
            self.gap = nn.AdaptiveAvgPool1d(1)
        
        if self.CLS_on:
            self.CLS = nn.Parameter(torch.zeros(1,3,1,1024))

        if self.key_value_emb is not None:
            if self.key_value_emb.lower()=='k':
                self.key_embedding = nn.Linear(in_features=1024,out_features=self.dim)
            elif self.key_value_emb.lower()=='v':
                self.value_embedding = nn.Linear(in_features=self.dim,out_features=self.dim)
            elif ''.join(sorted(self.key_value_emb.lower()))=='kv':
                self.key_embedding = nn.Linear(in_features=1024,out_features=self.dim)
                if self.model_name=='GoogleNet_Attention':
                    self.value_embedding = nn.Linear(in_features=1024,out_features=self.dim)
            else:
                raise

        if self.Layernorm:
            if self.Skip_connection=='KC':
                self.layernorm1 = nn.BatchNorm2d(num_features=1)
            elif self.Skip_connection=='CF':
                self.layernorm2 = nn.BatchNorm2d(num_features=1)
            elif self.Skip_connection=='IF':
                self.layernorm3 = nn.BatchNorm2d(num_features=1)
            elif self.Skip_connection is None:
                pass
            else:
                raise

    def forward(self, x):
        # Take the number of frames
        n_frame = x.shape[2]
        
        # Linear projection if using CLS token as transformer ways
        if self.Positional_encoding_way=='Transformer':
            x = self.Positional_encoding_embedding(x)
        # Stack CLS token
        if self.CLS_on:
            x = torch.cat((self.CLS,x),dim=2)
            CT_adjust = nn.AdaptiveAvgPool2d((n_frame,self.dim))
        
        # Positional encoding (Transformer ways)
        if self.Positional_encoding_way=='Transformer':
            if self.Positional_encoding is not None:
                x = self.Positional_encoding_op(x)
            # Dropout (Transformer ways)
            if self.Dropout_on:
                x = self.dropout(x)
        elif self.Positional_encoding_way=='PGL_SUM' or self.Positional_encoding_way is None:
            pass
        else:
            raise

        # Key Embedding
        if self.key_value_emb is not None and self.key_value_emb.lower() in ['k','kv']:
            key = self.key_embedding(x)
        elif self.key_value_emb is None:
            key = x
        else:
            raise
        
        # CNN as attention algorithm
        x_att = self.googlenet(key)

        # Skip connection (KC)
        if self.Skip_connection is not None:
            if self.Skip_connection=='KC':
                x_att = x_att + key.squeeze(0)[0]
                if self.Layernorm:
                    x_att = self.layernorm1(x_att.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
            elif self.Skip_connection in ['CF','IF']:
                pass
            else:
                raise
        elif self.Skip_connection is None:
            pass
        else:
            raise
        
        # Combine CLS token (CNN)
        if self.CLS_on:
            if self.CLS_mix=='CNN':
                x_att = CT_adjust(x_att.unsqueeze(0)).squeeze(0)
                x = CT_adjust(x.squeeze(0)).unsqueeze(0)
            elif self.CLS_mix in ['SM','Final']:
                pass
            else:
                raise
        else:
            pass
        
        # Scaling factor
        if self.Scale is not None:
            if self.Scale=='D':
                scaling_factor = x_att.shape[1]
            elif self.Scale=='T':
                scaling_factor = x_att.shape[0]
            elif self.Scale=='T_D':
                scaling_factor = x_att.shape[0] * x_att.shape[1]
            else:
                raise
            scaling_factor = scaling_factor ** 0.5
            x_att = x_att / scaling_factor
        elif self.Scale is None:
            pass
        
        # Positional encoding (PGL-SUM ways)
        if self.Positional_encoding_way=='PGL_SUM':
            if self.Positional_encoding is not None:
                x_att = self.Positional_encoding_op(x_att)
        elif self.Positional_encoding_way=='Transformer' or self.Positional_encoding_way is None:
            pass
        else:
            raise
        
        # softmax_axis
        x = x.squeeze(0)[0]
        if self.Softmax_axis=='T':
            temporal_attention = F.softmax(x_att,dim=0)
        elif self.Softmax_axis=='D':
            spatial_attention = F.softmax(x_att,dim=1)
        elif self.Softmax_axis=='TD':
            temporal_attention = F.softmax(x_att,dim=0)
            spatial_attention = F.softmax(x_att,dim=1)
        elif self.Softmax_axis is None:
            pass
        else:
            raise
        
        # Combine CLS token for softmax outputs (SM)
        if self.CLS_on:
            if self.CLS_mix=='SM':
                if self.Softmax_axis=='T':
                    temporal_attention = CT_adjust(temporal_attention.unsqueeze(0)).squeeze(0)
                elif self.Softmax_axis=='D':
                    spatial_attention = CT_adjust(spatial_attention.unsqueeze(0)).squeeze(0)
                elif self.Softmax_axis=='TD':
                    temporal_attention = CT_adjust(temporal_attention.unsqueeze(0)).squeeze(0)
                    spatial_attention = CT_adjust(spatial_attention.unsqueeze(0)).squeeze(0)
                elif self.Softmax_axis is None:
                    pass
                else:
                    raise
            elif self.CLS_mix in ['CNN','Final']:
                pass
            else:
                raise
        else:
            pass
        
        # Dropout (PGL-SUM ways)
        if self.Dropout_on and self.Positional_encoding_way=='PGL_SUM':
            if self.Softmax_axis=='T':
                temporal_attention = self.dropout(temporal_attention)
            elif self.Softmax_axis=='D':
                spatial_attention = self.dropout(spatial_attention)
            elif self.Softmax_axis=='TD':
                temporal_attention = self.dropout(temporal_attention)
                spatial_attention = self.dropout(spatial_attention)
            elif self.Softmax_axis is None:
                pass
            else:
                raise
        
        # Value Embedding
        if self.key_value_emb is not None and self.key_value_emb.lower() in ['v','kv']:
            if self.model_name=='GoogleNet_Attention':
                x_out = self.value_embedding(x)
            elif self.model_name=='GoogleNet':
                x_out = x_att
            else:
                raise
        elif self.key_value_emb is None:
            if self.model_name=='GoogleNet':
                x_out = x_att
            elif self.model_name=='GoogleNet_Attention':
                x_out = x
            else:
                raise
        else:
            raise
        
        # Combine CLS token for CNN outputs (SM)
        if self.CLS_on:
            if self.CLS_mix=='SM':
                x_out = CT_adjust(x_out.unsqueeze(0)).squeeze(0)

        # Apply Attention maps to input frame features
        if self.Softmax_axis=='T':
            x_out = x_out * temporal_attention
        elif self.Softmax_axis=='D':
            x_out = x_out * spatial_attention
        elif self.Softmax_axis=='TD':
            T,D = x_out.shape
            adjust_frame = T/D
            adjust_dimension = D/T
            if self.Balance=='T':
                x_out = x_out * temporal_attention * adjust_frame + x_out * spatial_attention
            elif self.Balance=='D':
                x_out = x_out * temporal_attention + x_out * spatial_attention * adjust_dimension
            elif self.Balance=='BD':
                if T>D:
                    x_out = x_out * temporal_attention + x_out * spatial_attention * adjust_dimension
                elif T<D:
                    x_out = x_out * temporal_attention * adjust_frame + x_out * spatial_attention
                elif T==D:
                    x_out = x_out * temporal_attention + x_out * spatial_attention
            elif self.Balance=='BU':
                if T>D:
                    x_out = x_out * temporal_attention * adjust_frame + x_out * spatial_attention
                elif T<D:
                    x_out = x_out * temporal_attention + x_out * spatial_attention * adjust_dimension
                elif T==D:
                    x_out = x_out * temporal_attention + x_out * spatial_attention
            elif self.Balance is None:
                x_out = x_out * temporal_attention + x_out * spatial_attention
            else:
                raise
        elif self.Softmax_axis is None:
            x_out = x_out * x_att
        else:
            raise
        
        # Skip Connection (CF)
        if self.Skip_connection is not None:
            if self.Skip_connection=='CF':
                if x_out.shape != x_att.shape:
                    x_att = CT_adjust(x_att.unsqueeze(0)).squeeze(0)
                x_out = x_out + x_att
                if self.Layernorm:
                    x_out = self.layernorm2(x_out.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
            elif self.Skip_connection in ['KC','IF']:
                pass
            else:
                raise
        elif self.Skip_connection is None:
            pass
        else:
            raise
        
        # Skip Connection (IF)
        if self.Skip_connection is not None:
            if self.Skip_connection=='IF':
                x_out = x_out + x
                if self.Layernorm:
                    x_out = self.layernorm3(x_out.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
            elif self.Skip_connection in ['KC','CF']:
                pass
            else:
                raise
        elif self.Skip_connection is None:
            pass
        else:
            raise
        
        # Combine CLS token (Final)
        if self.CLS_on:
            if self.CLS_mix=='Final':
                x_out = CT_adjust(x_out.unsqueeze(0)).squeeze(0)
            elif self.CLS_mix in ['CNN','SM']:
                pass
            else:
                raise
        else:
            pass
        
        # Classifier
        if self.Classifier_on:
            x_out = self.linear1(x_out)
            x_out = self.linear2(x_out)
            x_out = x_out.squeeze()
        else:
            x_out = self.gap(x_out)
            x_out = x_out.squeeze()

        return x_out