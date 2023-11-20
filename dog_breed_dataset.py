import pandas as pd
from PIL import Image
import os
import numpy as np
from torch.utils.data import Dataset
from torchvision.io import read_image
from PIL import Image
import torch
from sklearn.preprocessing import LabelEncoder


class DogBreedDataset(Dataset):
    def __init__(self, label_file, img_folder, transform=None, target_transform=None, mode="img", features = None):
        self.num_breeds = 120
        self.img_folder = img_folder
        self.transform = transform
        self.target_transform = target_transform
        self.labels = pd.read_csv(label_file)
        self.label_file_name = label_file
        self.mode = mode
        self.features = features

    

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if self.mode == 'img':
            img_path = os.path.join(self.img_folder, self.labels.iloc[idx, 1] + '.jpg')
            # image = Image.open(img_path)
            image = Image.open(img_path)
            # image = image.permute(1, 2, 0)
        else:
            image = self.features[idx]
            print(self.features[0])

        label = self.labels.iloc[idx, 3]

        # print("LABEL: ", label, type(label))
        # print(image.shape)
        if self.transform:
            # print(self.transform)

            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        # print(image.shape)
        return image, label
        