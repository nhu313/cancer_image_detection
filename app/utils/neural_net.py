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
from torch.utils.data import DataLoader, TensorDataset


class Convoultion_NN(ImageLoad):
    def __init__(self, dataset_path: str, learning_rate: float = 0.001, batch_size: int = 32, 
                 input_channels: int = 3, architecture: str = "wide"):
        # Inherit from ImageLoad
        super().__init__(dataset_path)
        # Input shape defaults to (input_channels, height, width)
        self.input = (input_channels, self.size[0], self.size[1])
        self.number_of_labels = self.df['Label'].nunique()
        self.learning_rate = learning_rate
        self.input_channels = input_channels  # (RGB for us, change if B/W)
        self.batch_size = batch_size  # For speed/memory; changes epoch iterations as batch_size must complete dataset cycle
        self.architecture = architecture
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        self.df['EncodedLabel'] = self.label_encoder.fit_transform(self.df['Label'])
        self.number_of_labels = len(self.label_encoder.classes_)

        # Convert images and labels to tensors
        self.image_tensors, self.label_tensors = self.create_image_tensors()

        # Define model and parameters
        self.model = self.build_model(architecture=self.architecture).to(self.device)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def create_image_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Converts image data from the DataFrame into tensors and normalizes them.

        Returns:
            Tuple of image and label tensors.
        """
        image_tensors = []
        label_tensors = torch.tensor(self.df['EncodedLabel'].values, dtype=torch.long).to(self.device)

        for _, row in tqdm(self.df.iterrows(), len(self.df)):
            img_array = row['Image'].copy()  
            img_tensor = torch.tensor(img_array, dtype=torch.float32) / 255.0  # Normalize to [0, 1]
            img_tensor = F.normalize(img_tensor)  # Further normalization (optional)
            img_tensor = img_tensor.permute(2, 0, 1)  # Convert (H, W, C) to (C, H, W)
            image_tensors.append(img_tensor)

        image_tensors = torch.stack(image_tensors).to(self.device)
        print('Image and label tensors created')
        return image_tensors, label_tensors
    
    def build_model(self, architecture: str) -> nn.Sequential:
        """
        Builds the CNN model based on the specified architecture.

        Args:
            architecture (str): Architecture type ("wide").

        Returns:
            nn.Sequential: A sequential model based on the desired architecture.
        """
        layers = []

        if architecture == "wide":
            # Wide architecture
            layers.extend([
                # Convolution 1
                nn.Conv2d(self.input_channels, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout(0.25),
                # Convolution 2
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout(0.25),
                # Linear transformation
                nn.Flatten(),
                nn.Linear(256 * 16 * 16, 512),  # Adjust based on input size
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, self.number_of_labels)
            ])
        
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output predictions.
        """
        return self.model(x)
    
    def train_model(self, epochs: int):
        """
        Trains the CNN model for a specified number of epochs.

        Args:
            epochs (int): Number of training epochs.
        """
        self.model.train()
        # TODO research
        dataset = TensorDataset(self.image_tensors, self.label_tensors)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(epochs):
            epoch_loss = 0
            for img_tensors, label_tensors in tqdm(dataloader, total=len(dataloader)):
                img_tensors, label_tensors = img_tensors.to(self.device), label_tensors.to(self.device)
                # Forward pass
                output = self.model(img_tensors)  
                loss = self.loss_function(output, label_tensors)
                # Zero gradients
                self.optimizer.zero_grad()
                # Backpropagation  
                loss.backward()  
                # Update weights
                self.optimizer.step()  
                # loss
                epoch_loss += loss.item() 
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(dataloader):.4f}")

    def process_image(self, file_path: str) -> str:
        """
        Processes and predicts the label for a single image using the trained model.

        Args:
            file_path (str): Path to the image file.

        Returns:
            str: Predicted label.
        """
        self.model.eval()  # Set the model to evaluation mode
        
        with torch.no_grad():  # Disable gradient calculation for inference
            img_tensor = self.image_to_tensor(file_path)  # Convert image to tensor
            output = self.model(img_tensor.unsqueeze(0))  # Forward pass (add batch dimension)
            
            # Get the predicted label by finding the index of the max log-probability
            predicted_label_index = torch.argmax(output.data, dim=1).item()
            predicted_label = self.label_encoder.inverse_transform([predicted_label_index])[0]  # Convert back to label
            
            print("Model output:", output)  # Print the raw output for debugging
            return predicted_label
        
    def image_to_tensor(self, file_path: str = None, numpy_array: np.ndarray = None) -> torch.Tensor:
        """
        Turn numpy array or file_path into Tensor with normalized pixel values.

        Args:
            file_path (str, optional): Path to the image file.
            numpy_array (np.ndarray, optional): Numpy array representing the image.

        Returns:
            torch.Tensor: Normalized image tensor.
        """
        if file_path:
            image = self._open_img(file_path, add_noise=False)
        else:
            image = numpy_array

        image = image.copy()
        img_tensor = torch.tensor(image, dtype=torch.float32) / 255.0  # Normalize to [0, 1]
        img_tensor = F.normalize(img_tensor)  # Further normalization (optional)

        img_tensor = img_tensor.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        return img_tensor.to(self.device)  # Return tensor on correct device

if __name__ == "__main__":
    folder_path = "/Users/kjams/Desktop/research/health_informatics/app/data/testing_data"
    print('Initializing image loader...')
    images = ImageLoad(folder_path)
    # Prepare images for neural network
    images.main_loop()
    df = images.df  # Assuming df is now populated with image data
