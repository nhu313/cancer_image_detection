import os
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from load_images import ImageLoad  # Assuming ImageLoad is saved in load_images.py


class Convoultion_NN(ImageLoad):
    def __init__(self, dataset_path, learning_rate: float = 0.001, batch_size: int = 32, 
                 input_channels:int = 3, architecture: str = "wide"):
        # inherient from ImageLoad
        super().__init__(dataset_path)
        # C,W,H = Default is (3,64,64)
        self.input = (input_channels, self.size[0], self.size[1])
        self.number_of_labels = self.df['Label'].nunique()
        self.learning_rate = learning_rate
        self.input_channels = input_channels # (RGB for us change if B/W)
        # for speed/memory | changes epoch iterations as batch_size must complete dataset cycle
        self.batch_size = batch_size 
        self.architecture = architecture
        self.model = self.build_model(architecture='wide')
        # Define the loss function
        self.loss_function = nn.CrossEntropyLoss()
        # Move model to the appropriate device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        

    def build_model(self, architecture):
        """
        Builds the CNN model based on the specified architecture.

        Returns:
            nn.Sequential: A sequential model based on the desired architecture.
        """
        layers = []

        if architecture == "wide":
            # Wide architecture
            layers.extend([
                # One Convolution
                nn.Conv2d(
                    self.input_channels, 
                    128, # output
                    kernel_size=3, 
                    stride=1, 
                    padding=1),
                
                nn.BatchNorm2d(128),
                
                nn.ReLU(),
                
                nn.MaxPool2d(kernel_size=2, stride=2),
                
                
            ################## 2nd Convolution #################################
                
                
                nn.Conv2d(128, # input
                          256, # output
                          kernel_size=3, 
                          stride=1, 
                          padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),

                # Train
                nn.Flatten(),
                nn.Linear(256 * 16 * 16, 512),  # Adjust based on input size
                nn.ReLU(),
                nn.Linear(512, self.number_of_labels)
            ])
        
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor (3,64,64)

        Returns:
            torch.Tensor: Output predictions.
        """
        return self.model(x)
    
    def image_to_tensor(self, file_path: str):
        """
        Loads an image & converts to tensor.

        Args:
            file_path (str): Path to the image file.

        Returns:
            torch.Tensor: The transformed image tensor of shape (3, 64, 64) to match model
        """
        
        # Load & Resize the image
        image = self._open_img(file_path, add_noise=False)
        # Apply the transformation
        img_tensor = torch.tensor(image, dtype=torch.float32)  # Convert to float tensor
        # Normalize 
        img_tensor /= 255.0 
        # Permute from Numpy Array (H, W, C) to Tensor(C, H, W)
        img_tensor = img_tensor.permute(2, 0, 1)  
        return img_tensor









    def process_image(file):
        '''
        # TODO create neural net and process image lol
        '''

        return "200"


if __name__ == "__main__":
    folder_path = "/Users/kjams/Desktop/research/health_informatics/app/data/testing_data"
    print('init')
    images = ImageLoad(folder_path)
    # prepare images for neural network
    images.main_loop()
    df = images.df
