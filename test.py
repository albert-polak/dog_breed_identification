from torch.utils.data import DataLoader
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from torchvision import models
from dog_breed_dataset import DogBreedDataset

import lightning as L

from train import DogBreedModel, get_features

from torchvision.models import __dict__ as models_dict
from torchvision.models import Inception_V3_Weights
from torchvision.models import DenseNet121_Weights
from torchvision.models import ResNeXt50_32X4D_Weights

from torchvision import transforms

test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

use_gpu = torch.cuda.is_available()
# print("CUDA GPU: ", use_gpu)

def test():
    model = DogBreedModel

    ckpt_path = './models/models/resnet-model-epoch=5-val_loss=0.32-val_acc=0.90.ckpt'

    model = model.load_from_checkpoint(
    checkpoint_path=ckpt_path,
    map_location=None,
    )

    trainer = L.Trainer()
    test_dog_dataset = DogBreedDataset('./test.csv', './dog-breed-identification/imgs/', transform=test_transform)
    test_loader = DataLoader(test_dog_dataset, shuffle=False)

    inceptionv3_features, labels = get_features("inception_v3", test_loader, Inception_V3_Weights.IMAGENET1K_V1)
    densenet_features, _ = get_features("densenet121", test_loader, DenseNet121_Weights.IMAGENET1K_V1)
    resnext_features, _ = get_features("resnext50_32x4d", test_loader, ResNeXt50_32X4D_Weights.IMAGENET1K_V2)

    final_features = np.concatenate([inceptionv3_features,
                                 densenet_features,
                                 resnext_features,], axis=-1)
    print('Final feature maps shape', final_features.shape)
 
    feature_data = []
    for i in range(len(final_features)):
        feature_data.append([final_features[i], labels[i]])

    feature_loader = DataLoader(feature_data, batch_size=32)
    trainer.test(model=model, dataloaders=feature_loader)

if __name__ == "__main__":
    test()