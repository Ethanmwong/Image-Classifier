# Ensemble of ViT and EfficientNet for Image Classification

## Overview
This project involves the development of an image classification model using an ensemble of Vision Transformer (ViT) and EfficientNet. The model was trained on a limited dataset of 1000 images, with data augmentations applied to expand the dataset. The project utilized Sharpness Aware Minimization (SAM) for model generalization and Stochastic Gradient Descent for optimization. The model's performance was evaluated using accuracy and loss metrics. The project also explored the use of a weighted average ensemble method to further increase accuracy.

## Model
The model is an ensemble of ViT and EfficientNet. Specifically, we used ViT-Huge in conjunction with EfficientNet_v2_L. We discovered that freezing all layers except the last few layers was an effective way of teaching our model to learn different patterns from the limited training dataset we were provided. We applied this strategy to each pretrained model. To reduce overfitting, we added a dropout layer of 0.25. Despite this, overfitting was still a persistent issue. 

## Optimization
We used stochastic gradient descent enhanced with SAM to optimize the model’s parameters. It updates the weights using a fixed learning rate to minimize the loss function. A learning rate of 0.004 was chosen to ensure the model learns at a steady pace, reducing the risk of overshooting the minima. Additionally, we factored in a momentum of 0.9 to accelerate convergence and reach the global minimum efficiently. To classify the odds of an image being of a certain class, we used Cross-Entropy Loss from the pytorch library.

## Image Processing
For image processing, we converted the size of all training images into 480x480 to preserve details in the image as well as to standardize input size across all data. We normalized the images and used data augmentations, such as horizontal flips, color jittering, and vertical flips to expand our dataset, as we were only provided with 1000 images to train with. We partitioned the training folder into an 80 20 training and validation set using torch’s built in ImageFolder. We trained for many epochs with a batch size of 16. 

## Evaluation
To evaluate the performance of our model, we used accuracy as well as the training and validation loss to see if our model was overfitting or not. Graphs were helpful in visualizing whether or not our model is improving, or if it is overfitting or underfitting. 

## Usage
Make sure the dataset directory is structured as follows: The train folder should have 100 classes, each with 10 images. The test folder has 1000 random images. 
Ensure all dependencies are installed and the python version is at least 3.10. Also ensure files are in the same directory as the models
To run the models, open using Jupyter Notebook and run all cells.
The training files are Final_3_2.ipynb, final_sam_model.ipynb, ViT.ipynbm ViT_copy.ipynb. The Validation & Test file is ensemble.ipynb
Output will be saved in a folder called “predictions.csv”
