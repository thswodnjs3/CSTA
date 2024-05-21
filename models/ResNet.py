from functools import partial
from typing import Any, Callable, List, Optional, Type, TypeVar, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torchvision.transforms._presets import ImageClassification
from torchvision.utils import _log_api_usage_once
from torchvision.models._api import Weights, WeightsEnum
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models._utils import _ovewrite_named_param, handle_legacy_interface

from models.positional_encoding import FixedPositionalEncoding,LearnablePositionalEncoding,RelativePositionalEncoding,ConditionalPositionalEncoding

M = TypeVar("M", bound=nn.Module)

BUILTIN_MODELS = {}

def register_model(name: Optional[str] = None) -> Callable[[Callable[..., M]], Callable[..., M]]:
    def wrapper(fn: Callable[..., M]) -> Callable[..., M]:
        key = name if name is not None else fn.__name__
        if key in BUILTIN_MODELS:
            raise ValueError(f"An entry is already registered under the name '{key}'.")
        BUILTIN_MODELS[key] = fn
        return fn

    return wrapper

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# ResNet as attention
class ResNet_Att(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]] = BasicBlock,
        layers: List[int] = [2, 2, 2, 2],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor,n_frame) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        self.avgpool = nn.AdaptiveAvgPool2d((n_frame,1))
        x = self.avgpool(x)
        x = torch.squeeze(x)
        x = x.permute(1,0)

        return x

    def forward(self, x: Tensor) -> Tensor:
        n_frame = x.shape[2]
        return self._forward_impl(x,n_frame)
    
def _resnet(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> ResNet_Att:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNet_Att(block, layers, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model

_COMMON_META = {
    "min_size": (1, 1),
    "categories": _IMAGENET_CATEGORIES,
}

class ResNet18_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/resnet18-f37072fd.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 11689512,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnet",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 69.758,
                    "acc@5": 89.078,
                }
            },
            "_ops": 1.814,
            "_file_size": 44.661,
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        },
    )
    DEFAULT = IMAGENET1K_V1

@register_model()
@handle_legacy_interface(weights=("pretrained", ResNet18_Weights.IMAGENET1K_V1))
def resnet18(*, weights: Optional[ResNet18_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet_Att:
    """ResNet-18 from `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`__.

    Args:
        weights (:class:`~torchvision.models.ResNet18_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet18_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet18_Weights
        :members:
    """
    weights = ResNet18_Weights.verify(weights)

    return _resnet(BasicBlock, [2, 2, 2, 2], weights, progress, **kwargs)

# ResNet-based CSTA
class CSTA_ResNet(nn.Module):
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
                 Layernorm,dim=512):
        super().__init__()
        self.resnet = ResNet_Att()

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
                if self.model_name=='ResNet_Attention':
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

        x_att = self.resnet(key)

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
            if self.model_name=='ResNet_Attention':
                x_out = self.value_embedding(x)
            elif self.model_name=='ResNet':
                x_out = x_att
            else:
                raise
        elif self.key_value_emb is None:
            if self.model_name=='ResNet':
                x_out = x_att
            elif self.model_name=='ResNet_Attention':
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