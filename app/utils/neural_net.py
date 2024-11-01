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
        # encode string labels to ints happens in train method
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
        ################### One Convolution   ##################
                nn.Conv2d(
                    self.input_channels, 
                    128, # output
                    kernel_size=3, 
                    stride=1, 
                    padding=1),
                # Average Pooling
                nn.BatchNorm2d(128),
                # Activation Function
                nn.ReLU(),
                # Grab most important features
                nn.MaxPool2d(kernel_size=2, stride=2),
                
                
            ################## 2nd Convolution #################################
                
                
                nn.Conv2d(128, # input
                          256, # output
                          # sliding-window settings
                          kernel_size=3, 
                          stride=1, 
                          padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),

                # Linear transformation on input
                # y= xW^T +b
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
    
    def train(self, epochs: int):
        self.model.train()
        
        # Encode labels and save encoder for later use
        self.label_encoder = LabelEncoder()
        self.df['EncodedLabel'] = self.label_encoder.fit_transform(self.df['Label'])

        for epoch in range(epochs):
            epoch_loss = 0

            for _, row in tqdm(self.df.iterrows(), total=self.df.shape[0]):
                encoded_label = row['EncodedLabel']  
                image_array = row.iloc[1]  # Assuming the image data is in the second column
                
                img_tensor = self.image_to_tensor(numpy_array=image_array).unsqueeze(0)
                label_tensor = torch.tensor([encoded_label], dtype=torch.long, device=self.device)

                output = self.model(img_tensor)
                loss = self.loss_function(output, label_tensor)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                del img_tensor

            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(self.df):.4f}")

    def process_image(self, file_path: str):
        self.model.eval()
        
        with torch.no_grad():
            img_tensor = self.image_to_tensor(file_path).unsqueeze(0)
            output = self.model(img_tensor)
            predicted_label = torch.argmax(output, dim=1).item()
            predicted_label = self.label_encoder.inverse_transform([predicted_label])[0]  # Decode back to original label

        return predicted_label
    def image_to_tensor(self, file_path=None, numpy_array=None):
        if file_path:
            image = self._open_img(file_path, add_noise=False)
        else:
            image = numpy_array

        image = image.copy()
        img_tensor = torch.tensor(image, dtype=torch.float32) / 255.0
        img_tensor = img_tensor.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        
        return img_tensor.to(self.device)  # Return tensor on correct device

if __name__ == "__main__":
    folder_path = "/Users/kjams/Desktop/research/health_informatics/app/data/testing_data"
    print('init')
    images = ImageLoad(folder_path)
    # prepare images for neural network
    images.main_loop()
    df = images.df
