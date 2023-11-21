# dog_breed_identification

Dog breed classification model done as a way to practice pytorch lightning and deep learning. Inspired by https://github.com/yashrane/Dog-Breed-Identification/ and https://www.kaggle.com/code/khanrahim/dog-breed

## Description

The goal of this project is to create a model able to identify the breed of a dog in a picture. This is a multiclass classification problem with 120 different dog breeds. The model is trained on this Kaggle dataset(https://www.kaggle.com/competitions/dog-breed-identification). The model was trained using Google Colab's GPU with pytorch.

## Model architecture

The model is neural network that takes features obtained by 3 pretrained models as input and consists of a dropout layer and a fully connected layer to classify input as one of 120 classes. The pretrained models used are InceptionV3, Densenet121 and ResNeXt50_32X4D. The features obtained by the pretrained models are concatenated and fed to the dropout layer. The dropout layer is used to prevent overfitting. 

## Results

The model was trained for 5 epochs with a batch size of 64. The model achieved a validation accuracy of 90% and a test accuracy of 89%. The model is able to identify dog breeds with great accuracy.

