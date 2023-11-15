from torch.utils.data import DataLoader
import os
import torch
import torch.nn as nn
import pandas as pd

from torchvision import models
from dog_breed_dataset import DogBreedDataset

import lightning as L

# from train import ResNetModel
import train
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.augmentations.transforms import Normalize

use_gpu = torch.cuda.is_available()
# print("CUDA GPU: ", use_gpu)

def train_from_checkpoint():
    model = train.ResNetModel()

    ckpt_path = './models/models/last.ckpt'

    # model = model.load_from_checkpoint(
    # checkpoint_path=ckpt_path,
    # map_location=None,
    # )

    trainer = L.Trainer()
    print("COSSSSOKADOSKDPAKSOPKD")
    train_dog_dataset = DogBreedDataset('./train.csv', './dog-breed-identification/imgs/', transform=train.transform_a)
    train_loader = DataLoader(train_dog_dataset, batch_size=16)

    val_dog_dataset = DogBreedDataset('./val.csv', './dog-breed-identification/imgs/', transform=train.transform_val)
    val_loader = DataLoader(val_dog_dataset, batch_size=16)
    trainer.fit(model=model, ckpt_path=ckpt_path, train_dataloaders=train_loader, val_dataloaders=val_loader)
    

train_from_checkpoint()