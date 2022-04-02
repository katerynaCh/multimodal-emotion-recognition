# -*- coding: utf-8 -*-
"""
This code is base on https://github.com/okankop/Efficient-3DCNNs
"""

import torch
import torch.utils.data as data
from PIL import Image
import functools
import numpy as np
import librosa


def video_loader(video_dir_path):
    video = np.load(video_dir_path)    
    video_data = []
    for i in range(np.shape(video)[0]):
        video_data.append(Image.fromarray(video[i,:,:,:]))    
    return video_data

def get_default_video_loader():
    return functools.partial(video_loader)

def load_audio(audiofile, sr):
    audios = librosa.core.load(audiofile, sr)
    y = audios[0]
    return y, sr

def get_mfccs(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
    return mfcc

def make_dataset(subset, annotation_path):
    with open(annotation_path, 'r') as f:
        annots = f.readlines()
        
    dataset = []
    for line in annots:
        filename, audiofilename, label, trainvaltest = line.split(';')        
        if trainvaltest.rstrip() != subset:
            continue
        
        sample = {'video_path': filename,                       
                  'audio_path': audiofilename, 
                  'label': int(label)-1}
        dataset.append(sample)
    return dataset 
       

class RAVDESS(data.Dataset):
    def __init__(self,                 
                 annotation_path,
                 subset,
                 spatial_transform=None,
                 get_loader=get_default_video_loader, data_type = 'audiovisual', audio_transform=None):
        self.data = make_dataset(subset, annotation_path)
        self.spatial_transform = spatial_transform
        self.audio_transform=audio_transform
        self.loader = get_loader()
        self.data_type = data_type 

    def __getitem__(self, index):
        target = self.data[index]['label']
                

        if self.data_type == 'video' or self.data_type == 'audiovisual':        
            path = self.data[index]['video_path']
            clip = self.loader(path)
            
            if self.spatial_transform is not None:               
                self.spatial_transform.randomize_parameters()
                clip = [self.spatial_transform(img) for img in clip]            
            clip = torch.stack(clip, 0).permute(1, 0, 2, 3) 
            
            if self.data_type == 'video':
                return clip, target
            
        if self.data_type == 'audio' or self.data_type == 'audiovisual':
            path = self.data[index]['audio_path']
            y, sr = load_audio(path, sr=22050) 
            
            if self.audio_transform is not None:
                 self.audio_transform.randomize_parameters()
                 y = self.audio_transform(y)     
                 
            mfcc = get_mfccs(y, sr)            
            audio_features = mfcc 

            if self.data_type == 'audio':
                return audio_features, target
        if self.data_type == 'audiovisual':
            return audio_features, clip, target  

    def __len__(self):
        return len(self.data)
