import pandas as pd
from PIL import Image
import os
import numpy as np
from torch.utils.data import Dataset
from torchvision.io import read_image

class DogBreedDataset(Dataset):
    def __init__(self, label_file, img_folder, transform=None, target_transform=None):
        self.num_breeds = 120
        self.img_folder = img_folder
        self.transform = transform
        self.target_transform = target_transform
        self.labels = pd.read_csv(label_file)
        self.label_file_name = label_file

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_folder, self.labels.iloc[idx, 1] + '.jpg')
        # image = Image.open(img_path)
        image = read_image(img_path)
        image = image.permute(1, 2, 0)
        label = self.labels.iloc[idx, 0]
        print(image.shape)
        if self.transform:
            image = self.transform(image=np.array(image))
        if self.target_transform:
            label = self.target_transform(label)
        print(image.shape)
        return image, label