import csv
import numpy as np
import torch

from collections import Counter

from models.EfficientNet import CSTA_EfficientNet
from models.GoogleNet import CSTA_GoogleNet
from models.MobileNet import CSTA_MobileNet
from models.ResNet import CSTA_ResNet

# Count the number of parameters
def count_parameters(model,model_name):
    if model_name in ['GoogleNet','GoogleNet_Attention','ResNet','ResNet_Attention']:
        x = [param.numel() for name,param in model.named_parameters() if param.requires_grad and 'fc' not in name]
    elif model_name in ['EfficientNet','EfficientNet_Attention','MobileNet','MobileNet_Attention']:
        x = [param.numel() for name,param in model.named_parameters() if param.requires_grad and 'classifier' not in name]
    return sum(x) / (1024 * 1024)

# Funtion printing the number of parameters of models
def report_params(model_name,
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
    print(f"PARAMS: {count_parameters(model,model_name):.2f}M")

# Print all arguments and GPU setting
def print_args(args):
    print(args.kwargs)
    print(f"CUDA: {torch.version.cuda}")
    print(f"cuDNN: {torch.backends.cudnn.version()}")
    if 'cuda' in args.device:
        print(f"GPU: {torch.cuda.is_available()}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")

# Load ground truth for TVSum
def get_gt(dataset):
    if dataset=='TVSum':
        annot_path = f"./data/ydata-anno.tsv"
        with open(annot_path) as annot_file:
            annot = list(csv.reader(annot_file, delimiter="\t"))
        annotation_length = list(Counter(np.array(annot)[:, 0]).values())
        user_scores = []
        for idx in range(1,51):
            init = (idx - 1) * annotation_length[idx-1]
            till = idx * annotation_length[idx-1]
            user_score = []
            for row in annot[init:till]:
                curr_user_score = row[2].split(",")
                curr_user_score = np.array([float(num) for num in curr_user_score])
                curr_user_score = curr_user_score / curr_user_score.max(initial=-1)
                curr_user_score = curr_user_score[::15]

                user_score.append(curr_user_score)
            user_scores.append(user_score)
        return user_scores
    elif dataset=='SumMe':
        return None
    else:
        raise
