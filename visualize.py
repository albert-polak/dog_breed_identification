from torch.utils.data import DataLoader
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import models
from dog_breed_dataset import DogBreedDataset

from matplotlib import pyplot as plt

import lightning as L

from train import DogBreedModel, get_features

from torchvision.models import __dict__ as models_dict
from torchvision.models import Inception_V3_Weights
from torchvision.models import DenseNet121_Weights
from torchvision.models import ResNeXt50_32X4D_Weights

from torchvision import transforms

use_gpu = torch.cuda.is_available()

test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

def visualize(model, dataloader, class_names, num_images=6):
    num_img = 0
    for data in dataloader:
        inputs, labels, img_path = data

        features = []
        for i in range(len(inputs)):
            features.append([inputs[i], labels[i]])
        feature_loader = DataLoader(features, batch_size=32)

        inceptionv3_features, labels = get_features("inception_v3", feature_loader, Inception_V3_Weights.IMAGENET1K_V1)
        densenet_features, _ = get_features("densenet121", feature_loader, DenseNet121_Weights.IMAGENET1K_V1)
        resnext_features, _ = get_features("resnext50_32x4d", feature_loader, ResNeXt50_32X4D_Weights.IMAGENET1K_V2)
        
        final_features = np.concatenate([inceptionv3_features,
                                 densenet_features,
                                 resnext_features,], axis=-1)
        features = []
        for i in range(len(final_features)):
            features.append([final_features[i], labels[i]])

        feature_loader = DataLoader(features, batch_size=32)
        for data in feature_loader:
            input, label = data
            output = model(input)
            _, preds = torch.max(output.data, 1)

        for j in range(inputs.size()[0]):
            num_img += 1
            img_name = img_path[j]
            img = Image.open(img_name)
            plt.imshow(img),plt.title('predicted: {}'
                                      .format(class_names.loc[
                                          class_names['label'] == preds[j].item()]['breed']))
            print('wrote prediction#'+ str(num_img) )
            plt.savefig('predictions/prediction#' + str(num_img) + '.jpg')
            if num_img == num_images:
                return


if __name__ == "__main__":

    model = DogBreedModel

    ckpt_path = './models/models/resnet-model-epoch=5-val_loss=0.32-val_acc=0.90.ckpt'

    model = model.load_from_checkpoint(
    checkpoint_path=ckpt_path,
    map_location=None,
    )

    model.eval()

    labels = pd.read_csv("./train.csv")

    classes = (labels[['breed', 'label']])
    classes = classes.drop_duplicates(subset='breed')
    print(classes)
    
    test_dog_dataset = DogBreedDataset('./test.csv', './dog-breed-identification/imgs/', transform=test_transform)
    test_loader = DataLoader(test_dog_dataset, shuffle=False, batch_size=16)

    visualize(model, test_loader, classes)

