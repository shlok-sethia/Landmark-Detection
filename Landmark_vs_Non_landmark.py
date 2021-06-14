import timm
import pandas as pd
import warnings
from timm.models.efficientnet import efficientnet_l2

from timm.models.resnest import resnest269e, resnest50d_4s2x40d
from timm.models.resnet import resnext50_32x4d
warnings.filterwarnings('ignore')

import os
import sys
import glob
import sys
from pathlib import Path
root_path = Path(os.getcwd())
sys.path.append(str(root_path.parent))


sys.path.append('')
from tqdm import tqdm_notebook
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision.transforms import transforms
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
#from metric.metrics import accuracy

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import NeptuneLogger
from torchvision.transforms import transforms
from src.utils import load_config
from pl_bolts.callbacks import PrintTableMetricsCallback
from pytorch_lightning import Trainer
import argparse

cfg = load_config('config.yml')

def dict_to_args(d):

    args = argparse.Namespace()

    def dict_to_args_recursive(args, d, prefix=''):
        for k, v in d.items():
            if type(v) == dict:
                dict_to_args_recursive(args, v, prefix=k)
            elif type(v) in [tuple, list]:
                continue
            else:
                if prefix:
                    args.__setattr__(prefix + '_' + k, v)
                else:
                    args.__setattr__(k, v)
    dict_to_args_recursive(args, d)
    return args

checkpoint_callback = ModelCheckpoint(
    filepath= cfg['model_checkpoint_params']['models_save_path'],
    save_weights_only= cfg['model_checkpoint_params']['save_weights_only'],
    save_top_k= cfg['model_checkpoint_params']['save_top_k'],
    verbose= cfg['model_checkpoint_params']['verbose'],
    monitor= cfg['model_checkpoint_params']['monitor'],
    mode = cfg['model_checkpoint_params']['mode']
)

early_stop_callback = EarlyStopping(
    monitor = cfg['early_stop_params']['monitor'],
    min_delta = cfg['early_stop_params']['min_delta'],
    patience = cfg['early_stop_params']['patience'],
    verbose = cfg['early_stop_params']['verbose'],
    mode = cfg['early_stop_params']['mode']
)

neptune_logger = NeptuneLogger(
    api_key = cfg['api_params']['api_key'],
    project_name = cfg['api_params']['project_name'],
    params = cfg['logging_params'],
    experiment_name = cfg['api_params']['experiment_name'],
    close_after_fit = cfg['api_params']['close_after_fit']
) 

def data_prepare(landmark_path, non_landmark_path):
    label_path_dict = {}
    non_path_dict = {}
    label_dict = {}
    non_label_dict = {}
    for image_path in tqdm_notebook(landmark_path):
        image_id = image_path.split('/')[-1][:-4]
        label_dict[image_id] = 0
        label_path_dict[image_id] = image_path 

    for image_path in tqdm_notebook(non_landmark_path):
        image_id = image_path.split('/')[-1][:-4]
        non_label_dict[image_id] = 1
        non_path_dict[image_id] = image_path 

    label_path_dict.update(non_path_dict)
    label_dict.update(non_label_dict)

    df = pd.DataFrame(columns=['id', 'path', 'label'])
    df['id'] = list(label_path_dict.keys())
    df['path'] = df['id'].apply(lambda x: label_path_dict.get(x))
    df['label'] = df['id'].apply(lambda x: label_dict.get(x))
    return df

class LandvsNoLandDataset(Dataset):

    def __init__(self, df, mode):
        print(f'creating data loader - {mode}')
        assert mode in ['train', 'val', 'test']
        self.df = df
        self.mode = mode

        transforms_list = []

        if self.mode == 'train':
            transforms_list = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomChoice([
                    transforms.Resize((256, 256)),
                    transforms.RandomResizedCrop(224),
                    transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
                    transforms.RandomAffine(degrees=15, translate=(0.2, 0.2),
                                            scale=(0.8, 1.2), shear=15,
                                            resample=Image.BILINEAR)
                ])
            ]

        transforms_list.extend([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ])
        self.transforms = transforms.Compose(transforms_list)

def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int):
        image_id = self.df.loc[index, 'id']
        image_path = self.df.loc[index, 'path']
        label = self.df.loc[index, 'label']

        image = Image.open(image_path)
        image = image.convert('RGB')

        image = self.transforms(image)

        if self.mode == 'test':
            return image
        else:
            return image, label
