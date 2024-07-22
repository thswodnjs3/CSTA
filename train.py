import h5py
import numpy as np
import shutil
import torch

from tqdm import tqdm

from config import get_config
from dataset import create_dataloader
from evaluation_metrics import get_corr_coeff
from generate_summary import generate_summary
from model import set_model
from utils import report_params, print_args, get_gt

# Load configurations
config = get_config()

# Print information of setting
print_args(config)

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

# Start training
for dataset in tqdm(config.datasets,total=len(config.datasets),ncols=70,leave=True,position=0):
    user_scores = get_gt(dataset)

    if dataset=='SumMe':
        batch_size = 1 if config.batch_size=='1' else int(config.SumMe_len*0.8*float(config.batch_size))
    elif dataset=='TVSum':
        batch_size = 1 if config.batch_size=='1' else int(config.TVSum_len*0.8*float(config.batch_size))

    for split_id,(train_loader,test_loader) in tqdm(enumerate(create_dataloader(dataset)),total=5,ncols=70,leave=False,position=1,desc=dataset):
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
        model.to(config.device)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(),lr=float(config.learning_rate),weight_decay=float(config.weight_decay))

        model_selection_kendall = -1    
        model_selection_spear = -1

        for epoch in tqdm(range(config.epochs),total=config.epochs,ncols=70,leave=False,position=2,desc=f'Split{split_id+1}'):
            model.train()
            update_loss = 0.0
            batch = 0

            for feature,gtscore,dataset_name,video_num in tqdm(train_loader,ncols=70,leave=False,position=3,desc=f'Epoch{epoch+1}_TRAIN'):
                feature = feature.to(config.device)
                gtscore = gtscore.to(config.device)
                output = model(feature)

                loss = criterion(output,gtscore) 
                loss.requires_grad_(True)
                
                update_loss += loss
                batch += 1

                if batch==batch_size:
                    optimizer.zero_grad()
                    update_loss = update_loss / batch
                    update_loss.backward()
                    optimizer.step()
                    update_loss = 0.0
                    batch = 0

            if batch>0:
                optimizer.zero_grad()
                update_loss = update_loss / batch
                update_loss.backward()
                optimizer.step()
                update_loss = 0.0
                batch = 0

            val_spears = []
            val_kendalls = []
            model.eval()
            with torch.no_grad():
                for feature,gtscore,dataset_name,video_num in tqdm(test_loader,ncols=70,leave=False,position=3,desc=f'Epoch{epoch+1}_TEST'):
                    feature = feature.to(config.device)
                    gtscore = gtscore.to(config.device)
                    output = model(feature)

                    if dataset_name in ['SumMe','TVSum']:
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
                        
                        val_spears.append(spear)
                        val_kendalls.append(kendall)

            if np.mean(val_kendalls) > model_selection_kendall and np.mean(val_spears) > model_selection_spear:
                model_selection_kendall = np.mean(val_kendalls)
                model_selection_spear = np.mean(val_spears)
                torch.save(model.state_dict(), './tmp/weight.pt')
        shutil.move('./tmp/weight.pt', f'./weights/{dataset}/split{split_id+1}.pt')
