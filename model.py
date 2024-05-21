from torchvision.models import EfficientNet_B0_Weights,GoogLeNet_Weights,MobileNet_V2_Weights,ResNet18_Weights

from models.EfficientNet import CSTA_EfficientNet
from models.GoogleNet import CSTA_GoogleNet
from models.MobileNet import CSTA_MobileNet
from models.ResNet import CSTA_ResNet

# Load models depending on CNN
def set_model(model_name,
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
              Layernorm):
    if model_name in ['EfficientNet','EfficientNet_Attention']:
        model = CSTA_EfficientNet(
            model_name=model_name,
            Scale=Scale,
            Softmax_axis=Softmax_axis,
            Balance=Balance,
            Positional_encoding=Positional_encoding,
            Positional_encoding_shape=Positional_encoding_shape,
            Positional_encoding_way=Positional_encoding_way,
            Dropout_on=Dropout_on,
            Dropout_ratio=Dropout_ratio,
            Classifier_on=Classifier_on,
            CLS_on=CLS_on,
            CLS_mix=CLS_mix,
            key_value_emb=key_value_emb,
            Skip_connection=Skip_connection,
            Layernorm=Layernorm
        )
        state_dict = EfficientNet_B0_Weights.IMAGENET1K_V1.get_state_dict(progress=False)
        model.efficientnet.load_state_dict(state_dict)
    elif model_name in ['GoogleNet','GoogleNet_Attention']:
        model = CSTA_GoogleNet(
            model_name=model_name,
            Scale=Scale,
            Softmax_axis=Softmax_axis,
            Balance=Balance,
            Positional_encoding=Positional_encoding,
            Positional_encoding_shape=Positional_encoding_shape,
            Positional_encoding_way=Positional_encoding_way,
            Dropout_on=Dropout_on,
            Dropout_ratio=Dropout_ratio,
            Classifier_on=Classifier_on,
            CLS_on=CLS_on,
            CLS_mix=CLS_mix,
            key_value_emb=key_value_emb,
            Skip_connection=Skip_connection,
            Layernorm=Layernorm
        )
        state_dict = GoogLeNet_Weights.IMAGENET1K_V1.get_state_dict(progress=False)
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('aux')}
        new_state_dict = model.googlenet.state_dict()
        for name,param in state_dict.items():
            new_state_dict[name] = param
        model.googlenet.load_state_dict(new_state_dict)
    elif model_name in ['MobileNet','MobileNet_Attention']:
        model = CSTA_MobileNet(
            model_name=model_name,
            Scale=Scale,
            Softmax_axis=Softmax_axis,
            Balance=Balance,
            Positional_encoding=Positional_encoding,
            Positional_encoding_shape=Positional_encoding_shape,
            Positional_encoding_way=Positional_encoding_way,
            Dropout_on=Dropout_on,
            Dropout_ratio=Dropout_ratio,
            Classifier_on=Classifier_on,
            CLS_on=CLS_on,
            CLS_mix=CLS_mix,
            key_value_emb=key_value_emb,
            Skip_connection=Skip_connection,
            Layernorm=Layernorm
        )
        state_dict = MobileNet_V2_Weights.IMAGENET1K_V1.get_state_dict(progress=False)
        model.mobilenet.load_state_dict(state_dict)
    elif model_name in ['ResNet','ResNet_Attention']:
        model = CSTA_ResNet(
            model_name=model_name,
            Scale=Scale,
            Softmax_axis=Softmax_axis,
            Balance=Balance,
            Positional_encoding=Positional_encoding,
            Positional_encoding_shape=Positional_encoding_shape,
            Positional_encoding_way=Positional_encoding_way,
            Dropout_on=Dropout_on,
            Dropout_ratio=Dropout_ratio,
            Classifier_on=Classifier_on,
            CLS_on=CLS_on,
            CLS_mix=CLS_mix,
            key_value_emb=key_value_emb,
            Skip_connection=Skip_connection,
            Layernorm=Layernorm
        )
        state_dict = ResNet18_Weights.IMAGENET1K_V1.get_state_dict(progress=False)
        model.resnet.load_state_dict(state_dict)
    else:
        raise
    return model