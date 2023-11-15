from torch.utils.data import DataLoader
import os
import torch
import torch.nn as nn
import pandas as pd

from torchvision import models
from dog_breed_dataset import DogBreedDataset

import lightning as L

from train import ResNetModel
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.augmentations.transforms import Normalize

test_transform = A.Compose([
    Normalize(),
    ToTensorV2()
])

use_gpu = torch.cuda.is_available()
# print("CUDA GPU: ", use_gpu)

def test():
    model = ResNetModel

    ckpt_path = './models/models/last.ckpt'

    model = model.load_from_checkpoint(
    checkpoint_path=ckpt_path,
    map_location=None,
    )

    trainer = L.Trainer()
    print("COSSSSOKADOSKDPAKSOPKD")
    test_dog_dataset = DogBreedDataset('./test.csv', './dog-breed-identification/imgs/', transform=test_transform)
    test_loader = DataLoader(test_dog_dataset)

    trainer.test(model=model, dataloaders=test_loader)

test()