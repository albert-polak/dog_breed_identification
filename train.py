from torch.utils.data import DataLoader
import os
import torch
import torch.nn as nn
import pandas as pd
# import torch.optim as optim

# import torch.nn.functional as F
from torchvision import models
from dog_breed_dataset import DogBreedDataset


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

class ResNetModel(L.LightningModule):
    def __init__(self, lr=1e-3, batch_size=16):
        super().__init__()
        # init a pretrained resnet
        self.model = models.resnet18(pretrained=True)


        self.lr = lr
        self.batch_size = batch_size
        
        self.num_classes = 120

        self.loss_fn = (
            nn.BCEWithLogitsLoss() if self.num_classes == 1 else nn.CrossEntropyLoss()
        )
        self.acc = Accuracy(
            task="binary" if self.num_classes == 1 else "multiclass", num_classes=self.num_classes
        )

        self.optimizer = Adam

        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)


    def forward(self, X):
        # X = X['image']
        # X = X.permute(0, 3, 1, 2)
        X = X.to(torch.float32).cuda()
        return self.model(X)

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.lr)

    def _step(self, batch):
        x, y = batch
        preds = self(x)

        if self.num_classes == 1:
            preds = preds.flatten()
            y = y.float()
        # if not torch.is_tensor(y):
        #     y = torch.tensor(y)
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

    model = ResNetModel()

    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        dirpath=save_path,
        filename="resnet-model-{epoch}-{val_loss:.2f}-{val_acc:0.2f}",
        monitor="val_loss",
        mode="min",
        save_last=True,
    )
    trainer_args = {
        # "accelerator": "gpu",
        # "devices": [0],
        "max_epochs": 25,
        "callbacks": [checkpoint_callback],
        "precision": 32,
    }

    train_dog_dataset = DogBreedDataset('./train.csv', './dog-breed-identification/imgs/', transform=transform_a)
    train_loader = DataLoader(train_dog_dataset, batch_size=16)

    val_dog_dataset = DogBreedDataset('./val.csv', './dog-breed-identification/imgs/', transform=transform_val)
    val_loader = DataLoader(val_dog_dataset, batch_size=16)

    trainer = L.Trainer(**trainer_args)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

train()
# labels = pd.read_csv('./train.csv')
# # img_path = os.path.join("./dog-breed-identification/", labels.iloc[0, 0])
# print(labels.iloc[0, 1])