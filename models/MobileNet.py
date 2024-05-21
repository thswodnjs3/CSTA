from functools import partial
from typing import Any, Callable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor

from torchvision.ops.misc import Conv2dNormActivation
from torchvision.transforms._presets import ImageClassification
from torchvision.utils import _log_api_usage_once
from torchvision.models._api import Weights, WeightsEnum
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models._utils import _make_divisible, _ovewrite_named_param, handle_legacy_interface

from models.positional_encoding import FixedPositionalEncoding,LearnablePositionalEncoding,RelativePositionalEncoding,ConditionalPositionalEncoding

class InvertedResidual(nn.Module):
    def __init__(
        self, inp: int, oup: int, stride: int, expand_ratio: int, norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        self.stride = stride
        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 instead of {stride}")

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(
                Conv2dNormActivation(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6)
            )
        layers.extend(
            [
                # dw
                Conv2dNormActivation(
                    hidden_dim,
                    hidden_dim,
                    stride=stride,
                    groups=hidden_dim,
                    norm_layer=norm_layer,
                    activation_layer=nn.ReLU6,
                ),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                norm_layer(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

# MobileNet as attention
class MobileNet_Att(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout: float = 0.2,
    ) -> None:
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use
            dropout (float): The droupout probability

        """
        super().__init__()
        _log_api_usage_once(self)

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError(
                f"inverted_residual_setting should be non-empty or a 4-element list, got {inverted_residual_setting}"
            )

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features: List[nn.Module] = [
            Conv2dNormActivation(3, input_channel, stride=2, norm_layer=norm_layer, activation_layer=nn.ReLU6)
        ]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel
        # building last several layers
        features.append(
            Conv2dNormActivation(
                input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6
            )
        )
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor, n_frame) -> Tensor:
        x = self.features(x)
        self.avgpool = nn.AdaptiveAvgPool2d((n_frame,1))
        x = self.avgpool(x)
        x = torch.squeeze(x)
        x = x.permute(1,0)
        return x

    def forward(self, x: Tensor) -> Tensor:
        n_frame = x.shape[2]
        return self._forward_impl(x, n_frame)
    
_COMMON_META = {
    "num_params": 3504872,
    "min_size": (1, 1),
    "categories": _IMAGENET_CATEGORIES,
}

class MobileNet_V2_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/mobilenet_v2-b0353104.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#mobilenetv2",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 71.878,
                    "acc@5": 90.286,
                }
            },
            "_ops": 0.301,
            "_file_size": 13.555,
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        },
    )
    IMAGENET1K_V2 = Weights(
        url="https://download.pytorch.org/models/mobilenet_v2-7ebf99e0.pth",
        transforms=partial(ImageClassification, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe-with-reg-tuning",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 72.154,
                    "acc@5": 90.822,
                }
            },
            "_ops": 0.301,
            "_file_size": 13.598,
            "_docs": """
                These weights improve upon the results of the original paper by using a modified version of TorchVision's
                `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V2

@handle_legacy_interface(weights=("pretrained", MobileNet_V2_Weights.IMAGENET1K_V1))
def mobilenet_v2(
    *, weights: Optional[MobileNet_V2_Weights] = None, progress: bool = True, **kwargs: Any
) -> MobileNet_Att:
    """MobileNetV2 architecture from the `MobileNetV2: Inverted Residuals and Linear
    Bottlenecks <https://arxiv.org/abs/1801.04381>`_ paper.

    Args:
        weights (:class:`~torchvision.models.MobileNet_V2_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.MobileNet_V2_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.mobilenetv2.MobileNetV2``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.MobileNet_V2_Weights
        :members:
    """
    weights = MobileNet_V2_Weights.verify(weights)

    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = MobileNet_Att(**kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model

# MobileNet-based CSTA
class CSTA_MobileNet(nn.Module):
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
        self.mobilenet = MobileNet_Att()

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
                if self.model_name=='MobileNet_Attention':
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

        x_att = self.mobilenet(key)

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
            if self.model_name=='MobileNet_Attention':
                x_out = self.value_embedding(x)
            elif self.model_name=='MobileNet':
                x_out = x_att
            else:
                raise
        elif self.key_value_emb is None:
            if self.model_name=='MobileNet':
                x_out = x_att
            elif self.model_name=='MobileNet_Attention':
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