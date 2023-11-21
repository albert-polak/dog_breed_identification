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

from train import ResNetModel, get_features

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


def predict_image(model, image_org, class_names):
    # plt.imshow(image_org)
    # plt.show()
    image = test_transform(image_org)

    img_dataloader = DataLoader([[image, 1]])

    inceptionv3_features, _ = get_features("inception_v3", img_dataloader, Inception_V3_Weights.IMAGENET1K_V1)
    densenet_features, _ = get_features("densenet121", img_dataloader, DenseNet121_Weights.IMAGENET1K_V1)
    resnext_features, _ = get_features("resnext50_32x4d", img_dataloader, ResNeXt50_32X4D_Weights.IMAGENET1K_V2)
    
    final_features = np.concatenate([inceptionv3_features,
                                 densenet_features,
                                 resnext_features,], axis=-1)
    # features = []
    # for i in range(len(final_features)):
    #     features.append([final_features[i], 1])
    feature_loader = DataLoader([[final_features[0], 1]])

    for data in feature_loader:
        input, label = data

        output = model(input)
        print(output.data)
        _, pred = torch.max(output.data, 1)

    print(pred)
    print(class_names.loc[class_names['label'] == pred.item()]['breed'])
    img = image_org
    plt.imshow(img),plt.title('predicted: {}'
                                .format(class_names.loc[
                                    class_names['label'] == pred.item()]['breed']))
    plt.savefig('predictions/' + img_path[27:])


if __name__ == "__main__":

    img_path = "./dog-breed-identification/mafi.jpg"
    # img_path = "./dog-breed-identification/imgs/000bec180eb18c7604dcecc8fe0dba07.jpg"
    image = Image.open(img_path)

    model = ResNetModel

    ckpt_path = './models/models/resnet-model-epoch=5-val_loss=0.32-val_acc=0.90.ckpt'

    model = model.load_from_checkpoint(
    checkpoint_path=ckpt_path,
    map_location=None,
    )

    model.eval()

    labels = pd.read_csv("./train.csv")

    classes = (labels[['breed', 'label']])
    classes = classes.drop_duplicates(subset='breed')

    predict_image(model, image, classes)
