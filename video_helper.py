# Reference code: https://github.com/li-plus/DSNet/blob/1804176e2e8b57846beb063667448982273fca89/src/helpers/video_helper.py
import cv2
import numpy as np
import torch
import torch.nn as nn

from os import PathLike
from pathlib import Path
from PIL import Image
from torchvision import transforms, models
from torchvision.models import GoogLeNet_Weights

from kts.cpd_auto import cpd_auto

class FeatureExtractor(object):
    def __init__(self, device):
        self.device = device
        self.transforms = GoogLeNet_Weights.IMAGENET1K_V1.transforms()
        weights = GoogLeNet_Weights.IMAGENET1K_V1
        self.model = models.googlenet(weights=weights)
        self.model = nn.Sequential(*list(self.model.children())[:-2])
        self.model.to(self.device)
        self.model.eval()

    def run(self, img: np.ndarray):
        img = Image.fromarray(img)
        img = self.transforms(img)
        batch = img.unsqueeze(0)
        with torch.no_grad():
            batch = batch.to(self.device)
            feat = self.model(batch)
            feat = feat.squeeze()

        assert feat.shape == (1024,), f'Invalid feature shape {feat.shape}: expected 1024'
        # normalize frame features
        feat = feat / (torch.norm(feat) + 1e-10)
        return feat
    
class VideoPreprocessor(object):
    def __init__(self, sample_rate: int, device: str):
        self.model = FeatureExtractor(device)
        self.sample_rate = sample_rate

    def get_features(self, video_path: PathLike):
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        assert cap is not None, f'Cannot open video: {video_path}'

        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        features = []
        n_frames = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            feat = self.model.run(frame)
            features.append(feat)
                
            n_frames += 1

        cap.release()
        features = torch.stack(features)
        return n_frames, features

    def kts(self, n_frames, features):
        seq_len = len(features)
        picks = np.arange(0, seq_len)

        # compute change points using KTS
        kernel = np.matmul(features.clone().detach().cpu().numpy(), features.clone().detach().cpu().numpy().T)
        change_points, _ = cpd_auto(kernel, seq_len - 1, 1, verbose=False)
        change_points = np.hstack((0, change_points, n_frames))
        begin_frames = change_points[:-1]
        end_frames = change_points[1:]
        change_points = np.vstack((begin_frames, end_frames - 1)).T
        return change_points, picks

    def run(self, video_path: PathLike):
        n_frames, features = self.get_features(video_path)
        cps, picks = self.kts(n_frames, features)
        return n_frames, features[::self.sample_rate,:], cps, picks[::self.sample_rate]
