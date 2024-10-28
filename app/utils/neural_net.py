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
from sklearn.preprocessing import LabelEncoder

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
        # encode string labels to ints
        

        self.model = self.build_model(architecture='wide')
        # Define the loss function
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

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
    
    def image_to_tensor(self, file_path: str = False, numpy_array:str = None):
        """
        Loads an image & converts to tensor.

        Args:
            file_path (str): Path to the image file.

        Returns:
            torch.Tensor: The transformed image tensor of shape (3, 64, 64) to match model
        """
        
        # Load & Resize the image
        if file_path != False:
            image = self._open_img(file_path, add_noise=False)
            # Apply the transformation
        else: 
            image = numpy_array
        
        image = image.copy()
        img_tensor = torch.tensor(image, dtype=torch.float32)  # Convert to float tensor
        # Normalize 
        img_tensor /= 255.0 
       
        # Permute from Numpy Array (H, W, C) to Tensor(C, H, W)
        img_tensor = img_tensor.permute(2, 0, 1)  
        del image 
        del img_tensor

        return img_tensor

    def train(self, epochs: int):
        
        """
        Train the CNN model.

        Args:
            epochs (int): Number of training epochs.
        """

        self.model.train()  # Set to train mode

        # Initialize LabelEncoder
        label_encoder = LabelEncoder()
        self.df['Label'] = label_encoder.fit_transform(self.df['Label'])

        for epoch in range(epochs):
            epoch_loss = 0

            for _, row in tqdm(self.df.iterrows(), total=self.df.shape[0]):
                label = row.iloc[0]  
                image_array = row.iloc[1]  
                
                # Convert the numpy array to a tensor
                img_tensor = self.image_to_tensor(numpy_array=image_array)

                # Add batch dimension
                img_tensor = img_tensor.unsqueeze(0)  # Shape becomes (1, C, H, W)

                # Create label tensor
                label_tensor = torch.tensor([label], dtype=torch.long)  # Create a single-element tensor

                # Forward pass
                output = self.model(img_tensor)

                # Check the shapes for debugging
               # print(f"Output shape: {output.shape}, Label shape: {label_tensor.shape}")

                # Compute loss
                loss = self.loss_function(output, label_tensor)
        
                # Backward pass and optimization
                self.optimizer.zero_grad()  # Clear gradients
                loss.backward()  # Backpropagation
                self.optimizer.step()  # Update weights

                epoch_loss += loss.item()
                
            
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(self.df):.4f}")


    def process_image(self, file):
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
