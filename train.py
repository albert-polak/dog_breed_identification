from torch.utils.data import DataLoader
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
# import torch.optim as optim

# import torch.nn.functional as F
from torchvision import models
from dog_breed_dataset import DogBreedDataset
from torchvision.models import __dict__ as models_dict
from torchvision.models import Inception_V3_Weights
from torchvision.models import DenseNet121_Weights
from torchvision.models import ResNeXt50_32X4D_Weights

from torchvision import transforms

from torchmetrics import Accuracy  

import lightning as L

from torch.optim import Adam

transform_a = transforms.Compose([
    transforms.RandomResizedCrop(size=(224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
])
transform_val = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
])


use_gpu = torch.cuda.is_available()
print("CUDA GPU: ", use_gpu)

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["TORCH_USE_CUDA_DSA"] = "1"

def get_features(model_name, data_loader, weights):
    # Prepare pipeline.
    data_transform = transforms.Compose([
        # transforms.Resize(input_size),

        weights.transforms(),
    ])

    # Create a DataLoader for the input data
    # dataset = torch.utils.data.TensorDataset(data)
    # data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Load pre-trained model
    model = models_dict[model_name](weights=weights)
    # model = nn.Sequential(*list(model.children())[:-1])  # Remove the fully connected layer
    model.fc = nn.Sequential()
    device = torch.device("cuda")
    model=model.to(device)

    # Extract features
    model.eval()
  
    labels = []
    feature_maps = []
    with torch.no_grad():
        for inputs in data_loader:
            # print(len(inputs[0]))
            inputs_transformed = data_transform(inputs[0])
            # print(inputs_transformed.shape)
            outputs = model(inputs_transformed.to(device))
            # print(outputs.shape)
            features = outputs.view(outputs.size(0), outputs.size(1), -1).mean(dim=2)
            # print(features.shape)
            feature_maps.append(features)
            labels.append(inputs[1])

    feature_maps = torch.cat(feature_maps, dim=0).to("cpu").numpy()
    labels = torch.cat(labels, dim=0).to("cpu").numpy()
    print('Feature maps shape: ', feature_maps.shape)
    return feature_maps, labels

class ResNetModel(L.LightningModule):
    def __init__(self, lr=1e-3, batch_size=32, input_shape=10):
        super().__init__()
        # init a pretrained resnet
        # self.model = models.resnet18(pretrained=True)
        
        self.num_classes = 120
        
        self.model = nn.Sequential(nn.Dropout(0.7),
                                   nn.Linear(input_shape[0], self.num_classes))

        self.lr = lr
        self.batch_size = batch_size
        
        self.loss_fn = nn.CrossEntropyLoss()
        
        self.acc = Accuracy(
            task="binary" if self.num_classes == 1 else "multiclass", num_classes=self.num_classes
        )

        self.optimizer = Adam(self.parameters(), lr=self.lr)

        # self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)


    def forward(self, X):
        # X = X['image']
        # X = X.permute(0, 3, 1, 2)
        X = X.to(torch.float32).cuda()
        return self.model(X)

    def configure_optimizers(self):
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.95)
        return {"optimizer": self.optimizer, "lr_scheduler": scheduler}

    def _step(self, batch):
        x, y = batch
        preds = self(x)

        loss = self.loss_fn(preds, y)
        # acc = self.acc(preds, y)
        acc_preds = torch.argmax(preds, dim=1)
        acc = self.acc(acc_preds, y)
        return loss, acc
    
    def training_step(self, batch, batch_idx):
        loss, acc = self._step(batch)
        # perform logging
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, acc = self._step(batch)
        # perform logging
        self.log("val_loss", loss, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        loss, acc = self._step(batch)
        # perform logging
        self.log("test_loss", loss, on_step=True, prog_bar=True, logger=True)
        self.log("test_acc", acc, on_step=True, prog_bar=True, logger=True)


def train():

    save_path = "./models"


    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        dirpath=save_path,
        filename="resnet-model-{epoch}-{val_loss:.2f}-{val_acc:0.2f}",
        monitor="val_loss",
        mode="min",
        save_last=True,
    )
    trainer_args = {
        "accelerator": "gpu",
        "devices": "auto",
        "strategy": "auto",
        "max_epochs": 25,
        "callbacks": [checkpoint_callback],
        "precision": 32,
    }

    train_dog_dataset = DogBreedDataset('./train.csv', './dog-breed-identification/imgs/', 
                                        transform=transform_a)
    train_loader = DataLoader(train_dog_dataset, batch_size=32, shuffle=False)

    val_dog_dataset = DogBreedDataset('./val.csv', './dog-breed-identification/imgs/',
                                      transform=transform_val)
    val_loader = DataLoader(val_dog_dataset, batch_size=32, shuffle=False)

    inceptionv3_features, labels = get_features("inception_v3", train_loader, Inception_V3_Weights.IMAGENET1K_V1)
    densenet_features, _ = get_features("densenet121", train_loader, DenseNet121_Weights.IMAGENET1K_V1)
    resnext_features, _ = get_features("resnext50_32x4d", train_loader, ResNeXt50_32X4D_Weights.IMAGENET1K_V2)

    inceptionv3_features_val, labels_val = get_features("inception_v3", val_loader, Inception_V3_Weights.IMAGENET1K_V1)
    densenet_features_val, _ = get_features("densenet121", val_loader, DenseNet121_Weights.IMAGENET1K_V1)
    resnext_features_val, _ = get_features("resnext50_32x4d", val_loader, ResNeXt50_32X4D_Weights.IMAGENET1K_V2)

    final_features = np.concatenate([inceptionv3_features,
                                 densenet_features,
                                 resnext_features,], axis=-1)
    print('Final feature maps shape', final_features.shape)
    
    final_features_val = np.concatenate([inceptionv3_features_val,
                                 densenet_features_val,
                                 resnext_features_val,], axis=-1)
    
    # train_dog_dataset = DogBreedDataset('./train.csv', './dog-breed-identification/imgs/', 
    #                                     mode="feature", features=final_features)
    # train_loader = DataLoader(train_dog_dataset, batch_size=32, shuffle=False)

    # val_dog_dataset = DogBreedDataset('./val.csv', './dog-breed-identification/imgs/', 
    #                                     mode="feature", features=final_features_val)
    # val_loader = DataLoader(val_dog_dataset, batch_size=32, shuffle=False)
    train_dataset = []
    for i in range(len(final_features)):
        train_dataset.append([final_features[i], labels[i]])

    val_dataset = []
    for i in range(len(final_features_val)):
        val_dataset.append([final_features_val[i], labels_val[i]])

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=32)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=32)

    model = ResNetModel(input_shape=final_features.shape[1:])


    trainer = L.Trainer(**trainer_args)
    trainer.fit(model=model, train_dataloaders=trainloader, val_dataloaders=valloader)

train()
# labels = pd.read_csv('./train.csv')
# # img_path = os.path.join("./dog-breed-identification/", labels.iloc[0, 0])
# print(labels.iloc[0, 1])