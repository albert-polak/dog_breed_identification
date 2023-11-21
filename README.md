# dog_breed_identification

Dog breed classification model done as a way to practice pytorch lightning and deep learning. Inspired by https://github.com/yashrane/Dog-Breed-Identification/ and https://www.kaggle.com/code/khanrahim/dog-breed

## Description

The goal of this project is to create a model able to identify the breed of a dog in a picture. This is a multiclass classification problem with 120 different dog breeds. The model is trained on this Kaggle dataset(https://www.kaggle.com/competitions/dog-breed-identification). The model was trained using Google Colab's GPU with pytorch.

## Model architecture

The model is neural network that takes features obtained by 3 pretrained models as input and consists of a dropout layer and a fully connected layer to classify input as one of 120 classes. The pretrained models used are InceptionV3, Densenet121 and ResNeXt50_32X4D. The features obtained by the pretrained models are concatenated and fed to the dropout layer. The dropout layer is used to prevent overfitting. 

## Results

The model was trained for 5 epochs with a batch size of 64. The model achieved a validation accuracy of 90% and a test accuracy of 89%. The model is able to identify dog breeds with great accuracy.

![mafi](https://github.com/albert-polak/dog_breed_identification/assets/80836949/46e06a10-cb07-46d4-9e23-83928b593635)
![prediction#1](https://github.com/albert-polak/dog_breed_identification/assets/80836949/851cf39d-a6ae-429c-ad76-af31f7b5efd5)
![prediction#2](https://github.com/albert-polak/dog_breed_identification/assets/80836949/aec92af3-9fbb-40ff-b889-f69cad573bb2)
![prediction#3](https://github.com/albert-polak/dog_breed_identification/assets/80836949/6c41cf28-d0fe-475b-a191-f317edd58100)
![000bec180eb18c7604dcecc8fe0dba07](https://github.com/albert-polak/dog_breed_identification/assets/80836949/5c756f81-9b93-4aa9-a542-75bf13e90dff)
![prediction#6](https://github.com/albert-polak/dog_breed_identification/assets/80836949/a906067e-b521-4136-8bc5-9c12236ca982)
