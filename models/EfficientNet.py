import copy
import math
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision.ops import StochasticDepth

from torchvision.ops.misc import Conv2dNormActivation, SqueezeExcitation
from torchvision.utils import _log_api_usage_once
from torchvision.models._utils import _make_divisible

from models.positional_encoding import FixedPositionalEncoding,LearnablePositionalEncoding,RelativePositionalEncoding,ConditionalPositionalEncoding

@dataclass
class _MBConvConfig:
    expand_ratio: float
    kernel: int
    stride: int
    input_channels: int
    out_channels: int
    num_layers: int
    block: Callable[..., nn.Module]

    @staticmethod
    def adjust_channels(channels: int, width_mult: float, min_value: Optional[int] = None) -> int:
        return _make_divisible(channels * width_mult, 8, min_value)
    
class MBConvConfig(_MBConvConfig):
    def __init__(
        self,
        expand_ratio: float,
        kernel: int,
        stride: int,
        input_channels: int,
        out_channels: int,
        num_layers: int,
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
        block: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        input_channels = self.adjust_channels(input_channels, width_mult)
        out_channels = self.adjust_channels(out_channels, width_mult)
        num_layers = self.adjust_depth(num_layers, depth_mult)
        if block is None:
            block = MBConv
        super().__init__(expand_ratio, kernel, stride, input_channels, out_channels, num_layers, block)

    @staticmethod
    def adjust_depth(num_layers: int, depth_mult: float):
        return int(math.ceil(num_layers * depth_mult))
    
class MBConv(nn.Module):
    def __init__(
        self,
        cnf: MBConvConfig,
        stochastic_depth_prob: float,
        norm_layer: Callable[..., nn.Module],
        se_layer: Callable[..., nn.Module] = SqueezeExcitation,
    ) -> None:
        super().__init__()

        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        activation_layer = nn.SiLU

        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            layers.append(
                Conv2dNormActivation(
                    cnf.input_channels,
                    expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        layers.append(
            Conv2dNormActivation(
                expanded_channels,
                expanded_channels,
                kernel_size=cnf.kernel,
                stride=cnf.stride,
                groups=expanded_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        )

        squeeze_channels = max(1, cnf.input_channels // 4)
        layers.append(se_layer(expanded_channels, squeeze_channels, activation=partial(nn.SiLU, inplace=True)))

        layers.append(
            Conv2dNormActivation(
                expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=None
            )
        )

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = cnf.out_channels

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        return result

class MBConvConfig(_MBConvConfig):
    def __init__(
        self,
        expand_ratio: float,
        kernel: int,
        stride: int,
        input_channels: int,
        out_channels: int,
        num_layers: int,
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
        block: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        input_channels = self.adjust_channels(input_channels, width_mult)
        out_channels = self.adjust_channels(out_channels, width_mult)
        num_layers = self.adjust_depth(num_layers, depth_mult)
        if block is None:
            block = MBConv
        super().__init__(expand_ratio, kernel, stride, input_channels, out_channels, num_layers, block)

    @staticmethod
    def adjust_depth(num_layers: int, depth_mult: float):
        return int(math.ceil(num_layers * depth_mult))
    
class FusedMBConvConfig(_MBConvConfig):
    def __init__(
        self,
        expand_ratio: float,
        kernel: int,
        stride: int,
        input_channels: int,
        out_channels: int,
        num_layers: int,
        block: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        if block is None:
            block = FusedMBConv
        super().__init__(expand_ratio, kernel, stride, input_channels, out_channels, num_layers, block)
    
class FusedMBConv(nn.Module):
    def __init__(
        self,
        cnf: FusedMBConvConfig,
        stochastic_depth_prob: float,
        norm_layer: Callable[..., nn.Module],
    ) -> None:
        super().__init__()

        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        activation_layer = nn.SiLU

        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            layers.append(
                Conv2dNormActivation(
                    cnf.input_channels,
                    expanded_channels,
                    kernel_size=cnf.kernel,
                    stride=cnf.stride,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

            layers.append(
                Conv2dNormActivation(
                    expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=None
                )
            )
        else:
            layers.append(
                Conv2dNormActivation(
                    cnf.input_channels,
                    cnf.out_channels,
                    kernel_size=cnf.kernel,
                    stride=cnf.stride,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = cnf.out_channels

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        return result
    
class FusedMBConvConfig(_MBConvConfig):
    def __init__(
        self,
        expand_ratio: float,
        kernel: int,
        stride: int,
        input_channels: int,
        out_channels: int,
        num_layers: int,
        block: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        if block is None:
            block = FusedMBConv
        super().__init__(expand_ratio, kernel, stride, input_channels, out_channels, num_layers, block)

def _efficientnet_conf(
    arch: str,
    **kwargs: Any,
) -> Tuple[Sequence[Union[MBConvConfig, FusedMBConvConfig]], Optional[int]]:
    inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]]
    if arch.startswith("efficientnet_b"):
        bneck_conf = partial(MBConvConfig, width_mult=kwargs.pop("width_mult"), depth_mult=kwargs.pop("depth_mult"))
        inverted_residual_setting = [
            bneck_conf(1, 3, 1, 32, 16, 1),
            bneck_conf(6, 3, 2, 16, 24, 2),
            bneck_conf(6, 5, 2, 24, 40, 2),
            bneck_conf(6, 3, 2, 40, 80, 3),
            bneck_conf(6, 5, 1, 80, 112, 3),
            bneck_conf(6, 5, 2, 112, 192, 4),
            bneck_conf(6, 3, 1, 192, 320, 1),
        ]
        last_channel = None
    elif arch.startswith("efficientnet_v2_s"):
        inverted_residual_setting = [
            FusedMBConvConfig(1, 3, 1, 24, 24, 2),
            FusedMBConvConfig(4, 3, 2, 24, 48, 4),
            FusedMBConvConfig(4, 3, 2, 48, 64, 4),
            MBConvConfig(4, 3, 2, 64, 128, 6),
            MBConvConfig(6, 3, 1, 128, 160, 9),
            MBConvConfig(6, 3, 2, 160, 256, 15),
        ]
        last_channel = 1280
    elif arch.startswith("efficientnet_v2_m"):
        inverted_residual_setting = [
            FusedMBConvConfig(1, 3, 1, 24, 24, 3),
            FusedMBConvConfig(4, 3, 2, 24, 48, 5),
            FusedMBConvConfig(4, 3, 2, 48, 80, 5),
            MBConvConfig(4, 3, 2, 80, 160, 7),
            MBConvConfig(6, 3, 1, 160, 176, 14),
            MBConvConfig(6, 3, 2, 176, 304, 18),
            MBConvConfig(6, 3, 1, 304, 512, 5),
        ]
        last_channel = 1280
    elif arch.startswith("efficientnet_v2_l"):
        inverted_residual_setting = [
            FusedMBConvConfig(1, 3, 1, 32, 32, 4),
            FusedMBConvConfig(4, 3, 2, 32, 64, 7),
            FusedMBConvConfig(4, 3, 2, 64, 96, 7),
            MBConvConfig(4, 3, 2, 96, 192, 10),
            MBConvConfig(6, 3, 1, 192, 224, 19),
            MBConvConfig(6, 3, 2, 224, 384, 25),
            MBConvConfig(6, 3, 1, 384, 640, 7),
        ]
        last_channel = 1280
    else:
        raise ValueError(f"Unsupported model type {arch}")

    return inverted_residual_setting, last_channel

# EfficientNet as attention
class EfficientNet_Att(nn.Module):
    def __init__(
        self,
        inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]] = _efficientnet_conf("efficientnet_b0", width_mult=1.0, depth_mult=1.0)[0],
        dropout: float = 0.2,
        stochastic_depth_prob: float = 0.2,
        num_classes: int = 1000,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        last_channel: Optional[int] = _efficientnet_conf("efficientnet_b0", width_mult=1.0, depth_mult=1.0)[1],
    ) -> None:
        """
        EfficientNet V1 and V2 main class

        Args:
            inverted_residual_setting (Sequence[Union[MBConvConfig, FusedMBConvConfig]]): Network structure
            dropout (float): The droupout probability
            stochastic_depth_prob (float): The stochastic depth probability
            num_classes (int): Number of classes
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
            last_channel (int): The number of channels on the penultimate layer
        """
        super().__init__()
        _log_api_usage_once(self)

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
            isinstance(inverted_residual_setting, Sequence)
            and all([isinstance(s, _MBConvConfig) for s in inverted_residual_setting])
        ):
            raise TypeError("The inverted_residual_setting should be List[MBConvConfig]")

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        layers: List[nn.Module] = []

        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(
            Conv2dNormActivation(
                3, firstconv_output_channels, kernel_size=3, stride=2, norm_layer=norm_layer, activation_layer=nn.SiLU
            )
        )

        total_stage_blocks = sum(cnf.num_layers for cnf in inverted_residual_setting)
        stage_block_id = 0
        for cnf in inverted_residual_setting:
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
    
                block_cnf = copy.copy(cnf)

                if stage:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1

                sd_prob = stochastic_depth_prob * float(stage_block_id) / total_stage_blocks

                stage.append(block_cnf.block(block_cnf, sd_prob, norm_layer))
                stage_block_id += 1

            layers.append(nn.Sequential(*stage))

        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = last_channel if last_channel is not None else 4 * lastconv_input_channels
        layers.append(
            Conv2dNormActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.SiLU,
            )
        )

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(lastconv_output_channels, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor,n_frame) -> Tensor:
        x = self.features(x)

        self.avgpool = nn.AdaptiveAvgPool2d((n_frame,1))
        x = self.avgpool(x)
        x = torch.squeeze(x)
        x = x.permute(1,0)

        return x

    def forward(self, x: Tensor) -> Tensor:
        n_frame = x.shape[2]
        return self._forward_impl(x,n_frame)

# EfficientNet-based CSTA
class CSTA_EfficientNet(nn.Module):
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
                 dim=1280):
        super().__init__()
        self.efficientnet = EfficientNet_Att()

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
                if self.model_name=='EfficientNet_Attention':
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
        n_frame = x.shape[2]
        
        if self.Positional_encoding_way=='Transformer':
            x = self.Positional_encoding_embedding(x)
        if self.CLS_on:
            x = torch.cat((self.CLS,x),dim=2)
            CT_adjust = nn.AdaptiveAvgPool2d((n_frame,self.dim))
        
        if self.Positional_encoding_way=='Transformer':
            if self.Positional_encoding is not None:
                x = self.Positional_encoding_op(x)
            if self.Dropout_on:
                x = self.dropout(x)
        elif self.Positional_encoding_way=='PGL_SUM' or self.Positional_encoding_way is None:
            pass
        else:
            raise

        if self.key_value_emb is not None and self.key_value_emb.lower() in ['k','kv']:
            key = self.key_embedding(x)
        elif self.key_value_emb is None:
            key = x
        else:
            raise

        x_att = self.efficientnet(key)

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
        
        if self.Positional_encoding_way=='PGL_SUM':
            if self.Positional_encoding is not None:
                x_att = self.Positional_encoding_op(x_att)
        elif self.Positional_encoding_way=='Transformer' or self.Positional_encoding_way is None:
            pass
        else:
            raise
        
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
        
        if self.key_value_emb is not None and self.key_value_emb.lower() in ['v','kv']:
            if self.model_name=='EfficientNet_Attention':
                x_out = self.value_embedding(x)
            elif self.model_name=='EfficientNet':
                x_out = x_att
            else:
                raise
        elif self.key_value_emb is None:
            if self.model_name=='EfficientNet':
                x_out = x_att
            elif self.model_name=='EfficientNet_Attention':
                x_out = x
            else:
                raise
        else:
            raise

        if self.CLS_on:
            if self.CLS_mix=='SM':
                x_out = CT_adjust(x_out.unsqueeze(0)).squeeze(0)

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
        
        if self.CLS_on:
            if self.CLS_mix=='Final':
                x_out = CT_adjust(x_out.unsqueeze(0)).squeeze(0)
            elif self.CLS_mix in ['CNN','SM']:
                pass
            else:
                raise
        else:
            pass

        if self.Classifier_on:
            x_out = self.linear1(x_out)
            x_out = self.linear2(x_out)
            x_out = x_out.squeeze()
        else:
            x_out = self.gap(x_out)
            x_out = x_out.squeeze()

        return x_out