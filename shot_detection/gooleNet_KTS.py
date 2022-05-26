from os import PathLike
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from numpy import linalg
from torch import nn
from torchvision import transforms
from shot_detection.modifiedGoogleNet import googlenet

from shot_detection.kts.cpd_auto import cpd_auto, seg_cpd_auto


class FeatureExtractor(object):
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.model = googlenet(pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[:-2])
        self.model = self.model.cuda().eval()


    def preprocess(self, img:np.ndarray) -> torch.tensor:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = self.transform(img)
        return img
    
    
    def run(self, img: np.ndarray) -> np.ndarray:
        img = Image.fromarray(img)
        img = self.transform(img)
        batch = img.unsqueeze(0)
        with torch.no_grad():
            feat = self.model(batch.cuda())
            feat = feat.squeeze().cpu().numpy()

        assert feat.shape == (1024,), f'Invalid feature shape {feat.shape}: expected 1024'
        # normalize frame features
        feat /= linalg.norm(feat) + 1e-10
        return feat
    
    def extract_feature_in_batch(self,data) -> np.ndarray:
        with torch.no_grad():
            feat = self.model(data.cuda())
            feat = feat.squeeze().cpu().numpy()

        if len(feat.shape) == 1:
            feat = np.reshape(feat, (1, feat.shape[0]))
        assert feat.shape[1] == 1024, f'Invalid feature shape {feat.shape}: expected 1024'
        # normalize frame features
        for i in range(feat.shape[0]):
            feat[i] /= linalg.norm(feat[i]) + 1e-10
        return feat


class VideoPreprocessor(object):
    def __init__(self, sample_rate: int) -> None:
        self.model = FeatureExtractor()
        self.sample_rate = sample_rate

    def get_features(self, video_path: PathLike):
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        assert cap is not None, f'Cannot open video: {video_path}'

        features = []
        n_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if n_frames % self.sample_rate == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                feat = self.model.run(frame)
                features.append(feat)

            n_frames += 1
        cap.release()
        
    
    def get_seg_features(self, video_cap, batch_size=10, max_len=3000):
        features = []
        n_frames = 0
        
        batch = []
        batch_num=0
        while True:
            ret, frame = video_cap.read()
            if not ret:
                break

            if n_frames % self.sample_rate == 0:
                img=self.model.preprocess(frame)
                batch.append(img)
                batch_num += 1
                if batch_num == batch_size:
                    data=torch.stack(batch)
                    feat = self.model.extract_feature_in_batch(data)
                    features.extend(feat.tolist())
                    batch.clear()
                    batch_num=0

            n_frames += 1
            if max_len>0 and n_frames>=max_len:
                break
        if batch_num>0:
            data = torch.stack(batch)
            feat = self.model.extract_feature_in_batch(data)
            features.extend(feat.tolist())
        features = np.array(features)
        return n_frames, features

    def kts(self, n_frames, features):
        seq_len = len(features)
        picks = np.arange(0, seq_len) * self.sample_rate

        # compute change points using KTS
        kernel = np.matmul(features, features.T)
   #     change_points, _ = cpd_auto(kernel, seq_len - 1, 1, verbose=False)
        change_points, _ = cpd_auto(kernel, seq_len - 1, 2, verbose=False)
        change_points *= self.sample_rate
        change_points = np.hstack((0, change_points, n_frames))
        begin_frames = change_points[:-1]
        end_frames = change_points[1:]
        change_points = np.vstack((begin_frames, end_frames - 1)).T

        n_frame_per_seg = end_frames - begin_frames
        return change_points, n_frame_per_seg, picks
    
    def seg_kts(self,features, max_overlap,vmax):
         seq_len = len(features)

         # compute change points using KTS
         kernel = np.matmul(features, features.T)
         change_points, scatter_mtx, overlap_len, last_scatter = seg_cpd_auto(kernel, seq_len - 1, vmax, max_overlap,desc_rate=self.sample_rate, verbose=False)
         return change_points, scatter_mtx, overlap_len, last_scatter
        

    def run(self, video_path: PathLike):
        n_frames, features = self.get_features(video_path)
        cps, nfps, picks = self.kts(n_frames, features)
        return n_frames, features, cps, nfps, picks
    
    def seg_shot_detect(self, video_path:PathLike, batch_size=100, max_len=3000, max_overlap=100,vmax=1.5):
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        assert cap is not None, f'Cannot open video: {video_path}'
        fps = round(cap.get(cv2.CAP_PROP_FPS))
        
        prev_features=np.empty(shape=[0,0])
        cps=[]
        last_overlap=0
        last_scatter=0
        t_frame_num=0
        t_seq_len=0
        while True:
            n_frames,seg_features = self.get_seg_features(cap,batch_size,max_len)
            t_frame_num += n_frames
            if n_frames<10:
                break
            if prev_features.shape[0]>0:
                features=np.concatenate((prev_features,seg_features),axis=0)
            else:
                features = seg_features
            change_points, scatter_mtx, overlap_len, end_scatter=self.seg_kts(features,max_overlap, vmax)
            if last_overlap==0:     #first segment, no overlapping with previous segment
                idx=features.shape[0]-overlap_len
                prev_features = features[idx:,:].copy()
                last_overlap=overlap_len
                last_scatter=end_scatter
                cps = change_points.tolist()
            else:
                start_pos=int(last_overlap/2)
                end_pos=min(last_overlap+start_pos,change_points[0]-1)
                curr_scatter=scatter_mtx[start_pos][end_pos]
                existing_len=t_seq_len-last_overlap
                change_points += existing_len    
                if curr_scatter<5.0  or curr_scatter<min((last_scatter*1.5), last_scatter+30):        #merge the segments
                    cps.extend(change_points.tolist())
                else:
                    cps.append(t_seq_len)
                    cps.extend(change_points.tolist())
                idx=features.shape[0]-overlap_len
                prev_features = features[idx:,:].copy()
                last_overlap=overlap_len
                last_scatter=end_scatter
            t_seq_len += len(seg_features)
        cap.release()
        cps=np.array(cps) 
        cps *= self.sample_rate
        cps = np.hstack((0, cps, t_frame_num))
        begin_frames = cps[:-1]
        end_frames = cps[1:]
        cps = np.vstack((begin_frames, end_frames - 1)).T
        n_frame_per_seg = end_frames - begin_frames
        picks = np.arange(0, t_seq_len) * self.sample_rate
        return t_frame_num, cps, n_frame_per_seg, picks

