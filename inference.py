import h5py
import numpy as np
import torch

from config import get_config
from dataset import create_dataloader
from evaluation_metrics import get_corr_coeff
from generate_summary import generate_summary
from model import set_model
from utils import report_params,get_gt

# Load configurations
config = get_config()

# Print the number of parameters
report_params(
    model_name=config.model_name,
    Scale=config.Scale,
    Softmax_axis=config.Softmax_axis,
    Balance=config.Balance,
    Positional_encoding=config.Positional_encoding,
    Positional_encoding_shape=config.Positional_encoding_shape,
    Positional_encoding_way=config.Positional_encoding_way,
    Dropout_on=config.Dropout_on,
    Dropout_ratio=config.Dropout_ratio,
    Classifier_on=config.Classifier_on,
    CLS_on=config.CLS_on,
    CLS_mix=config.CLS_mix,
    key_value_emb=config.key_value_emb,
    Skip_connection=config.Skip_connection,
    Layernorm=config.Layernorm
)

# Start testing
for dataset in config.datasets:
    user_scores = get_gt(dataset)
    split_kendalls = []
    split_spears = []

    for split_id,(train_loader,test_loader) in enumerate(create_dataloader(dataset)):
        model = set_model(
            model_name=config.model_name,
            Scale=config.Scale,
            Softmax_axis=config.Softmax_axis,
            Balance=config.Balance,
            Positional_encoding=config.Positional_encoding,
            Positional_encoding_shape=config.Positional_encoding_shape,
            Positional_encoding_way=config.Positional_encoding_way,
            Dropout_on=config.Dropout_on,
            Dropout_ratio=config.Dropout_ratio,
            Classifier_on=config.Classifier_on,
            CLS_on=config.CLS_on,
            CLS_mix=config.CLS_mix,
            key_value_emb=config.key_value_emb,
            Skip_connection=config.Skip_connection,
            Layernorm=config.Layernorm
        )
        model.load_state_dict(torch.load(f'./weights/{dataset}/split{split_id+1}.pt', map_location='cpu'))
        model.to(config.device)
        model.eval()

        kendalls = []
        spears = []
        with torch.no_grad():
            for feature,_,dataset_name,video_num in test_loader:
                feature = feature.to(config.device)
                output = model(feature)

                with h5py.File(f'./data/eccv16_dataset_{dataset_name.lower()}_google_pool5.h5','r') as hdf:
                    user_summary = np.array(hdf[video_num]['user_summary'])
                    sb = np.array(hdf[f"{video_num}/change_points"])
                    n_frames = np.array(hdf[f"{video_num}/n_frames"])
                    positions = np.array(hdf[f"{video_num}/picks"])
                scores = output.squeeze().clone().detach().cpu().numpy().tolist()
                summary = generate_summary([sb], [scores], [n_frames], [positions])[0]

                if dataset_name=='SumMe':
                    spear,kendall = get_corr_coeff([summary],[video_num],dataset_name,user_summary)
                elif dataset_name=='TVSum':
                    spear,kendall = get_corr_coeff([scores],[video_num],dataset_name,user_scores)
                
                spears.append(spear)
                kendalls.append(kendall)
        split_kendalls.append(np.mean(kendalls))
        split_spears.append(np.mean(spears))
        print("[Split{}]Kendall:{:.3f}, Spear:{:.3f}".format(
            split_id,split_kendalls[split_id],split_spears[split_id]
        ))
    print("[FINAL - {}]Kendall:{:.3f}, Spear:{:.3f}".format(
        dataset,np.mean(split_kendalls),np.mean(split_spears)
    ))
    print()
