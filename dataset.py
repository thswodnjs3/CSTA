import h5py
import torch

from torch.utils.data import Dataset,DataLoader

# Load split
def load_split(dataset):
    outputs = []
    with open(f'./splits/{dataset}_splits.txt','r') as f:
        lines = f.readlines()
        for line in lines:
            _,_,train_videos,test_videos = line.split('/')
            train_videos = train_videos.split(',')
            test_videos = test_videos.split(',')
            test_videos[-1] = test_videos[-1].replace('\n','')
            outputs.append((train_videos,test_videos))
    return outputs

# Create input,ground truth pair
def load_h5(videos,data_path,dataset_name):
    features = []
    gtscores = []
    dataset_names = []

    with h5py.File(data_path,'r') as hdf:
        for video in videos:
            feature = hdf[video]['features'][()]
            gtscore = hdf[video]['gtscore'][()]

            features.append(feature)
            gtscores.append(gtscore)
            dataset_names.append(dataset_name)
    return features,gtscores,dataset_names

# Create Dataset
class VSdataset(Dataset):
    def __init__(self,data,video_nums,transform=None):
        features,gtscores,dataset_names = data
        self.features = features
        self.gtscores = gtscores
        self.dataset_names = dataset_names
        self.video_nums = video_nums
        self.transform = transform
    def __len__(self):
        return len(self.video_nums)
    def __getitem__(self,idx):
        output_feature = torch.from_numpy(self.features[idx]).float()
        output_feature = output_feature.unsqueeze(0).expand(3,-1,-1)
        if self.transform is not None:
            output_feature=  self.transform(output_feature)
        return torch.unsqueeze(output_feature,0),torch.from_numpy(self.gtscores[idx]).float(),self.dataset_names[idx],self.video_nums[idx]
    
def collate_fn(sample):
    return sample[0]

# Create Dataloader
def create_dataloader(dataset):
    loaders = []
    
    splits = load_split(dataset=dataset)
    data_path = f'./data/eccv16_dataset_{dataset.lower()}_google_pool5.h5'

    for train_videos,test_videos in splits:
        train_data = load_h5(videos=train_videos,data_path=data_path,dataset_name=dataset)
        test_data = load_h5(videos=test_videos,data_path=data_path,dataset_name=dataset)

        train_dataset = VSdataset(data=train_data,video_nums=train_videos)
        test_dataset = VSdataset(data=test_data,video_nums=test_videos)
        train_loader = DataLoader(train_dataset,batch_size=1,shuffle=True,collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset,batch_size=1,shuffle=False,collate_fn=collate_fn)
        loaders.append((train_loader,test_loader))
    return loaders