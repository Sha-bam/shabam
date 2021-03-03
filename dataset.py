import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import numpy as np
import os


class GunshotForensicDataset(Dataset):
    '''
    Pulled from http://cadreforensics.com/
    '''

    def __init__(self, csv_path, path):
        self.folders = []
        self.file_names = []
        self.audio_files = []
        self.labels = []
        for folder in os.listdir(path):
            a = folder.split('_')
            gun_name = a[0]
            recording_method = a[1]

            for f in os.listdir(f'{path}/{folder}'):
                if f[0] != ".":
                    file_path = f'{path}/{folder}/{f}'
                    s = torchaudio.load(file_path)
                    print(s)
                    break

    def __getitem__(self, index):
        pass
