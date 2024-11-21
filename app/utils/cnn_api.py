import numpy as np
import torch
import torch.nn as nn
from PIL import Image



class CNN():
    def __init__(self, architecture: str = "wide", tensors: list = None, model_path: str = "cnn_model.pth"):
        '''
        tensors: list(image_tensors, label_tensors)
        model_path: str
        '''
        self.input_channels = 3
        self.number_of_labels = 4
        self.size = (128,128)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.build_model(architecture).to(self.device)
        self.batch_size = 32

        if tensors and len(tensors) == 2:

            self.load_model(model_path=model_path)
            self.image_tensors,self.label_tensors = self.load_tensors(tensors)
            print('CNN INIT SUCCESSFUL: 200')
        else:
            raise ValueError("Tensors must be provided as a list of two file paths.")

    def load_tensors(self, tensors):
        # Load saved tensors
        self.image_tensors = torch.load(tensors[0],weights_only=True).to(self.device)
        self.label_tensors = torch.load(tensors[1],weights_only=True).to(self.device)
        print("Tensors loaded from disk.")
        return self.image_tensors, self.label_tensors

    def build_model(self, architecture: str) -> nn.Sequential:
        """
        Builds the CNN model based on the specified architecture.

        Args:
            architecture (str): Architecture type ("deep-wide").

        Returns:
            nn.Sequential: A sequential model based on the desired architecture.
        """
        layers = []

        if architecture == "deep-wide":
            # Wide and deep architecture

            layers = [
                # Convolution Block 1
                nn.Conv2d(self.input_channels, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(negative_slope=0.01),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout2d(0.2),

                # Convolution Block 2
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),  # Corrected to 128 to match Conv2d output channels
                nn.LeakyReLU(negative_slope=0.01),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout2d(0.3),

                # Convolution Block 3
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(negative_slope=0.01),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout2d(0.5),

                # Global Average Pooling
                nn.AdaptiveAvgPool2d(1),  # Output will be 256 x 1 x 1

                # Fully Connected Layers
                nn.Flatten(),
                nn.Linear(256, 128),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Dropout(0.4),

                # Output Layer
                nn.Linear(128, self.number_of_labels)
            ]
        else:
            raise ValueError(f"Unsupported architecture type '{architecture}'")

        # Wrap layers in Sequential
        self.model = nn.Sequential(*layers)

        return self.model

    def load_model(self, model_path):
        # Load model parameters
        self.model.load_state_dict(torch.load(model_path, weights_only=True, map_location=self.device))
        self.model.eval()
        print("Model loaded from disk.")

    def _open_image(self, path_to_image:str):
        '''
        Return a resized NumpyArray of (128,128)
        '''
        img = Image.open(path_to_image)
        img = np.array(img)
        # Get the image dimensions
        height, width = img.shape[:2]
        # Crop out a portion of the bottom (e.g., remove an 1/8th of the height)
        new_height = height - (height // 8)
        img_cropped = img[:new_height, :]  # Crop from the top down to new_height
        # Resize the image using Pillow
        size = (width, new_height)  # Resize to the original width and the new height
        image_resized = Image.fromarray(img_cropped).resize(size)
        # Convert resized image back to a NumPy array
        img_resized = np.array(image_resized)
        return img_resized

    def predict_image(self, path_to_image) -> int:
        # Predict class for a single image tensor
        self.model.eval()
        img = self._open_image(path_to_image)
        img_tensor = torch.tensor(img, dtype=torch.float32) #/ 255.0  # Normalize to [0, 1]
        #img_tensor = F.normalize(img_tensor)

        img_tensor = img_tensor.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        img_tensor.to(self.device)  # Return tensor on correct device
        with torch.no_grad():
            output = self.model(img_tensor.unsqueeze(0))  # Add batch dimension
            predicted_label_index = torch.argmax(output, dim=1).item()
            return predicted_label_index

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output predictions.
        """
        return self.model(x)

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

        #image = image.copy()
        img_tensor = torch.tensor(image, dtype=torch.float32) #/ 255.0  # Normalize to [0, 1]
        #img_tensor = F.normalize(img_tensor)  # Further normalization (optional)

        img_tensor = img_tensor.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        return img_tensor.to(self.device)  # Return tensor on correct device

    def save_tensors(self):
        # Save tensors to file
        torch.save(self.image_tensors, "image_tensors.pt")
        torch.save(self.label_tensors, "label_tensors.pt")
        print("Tensors saved to disk.")


# Example usage
if __name__ == "__main__":
    tensor_paths = ["app/utils/data/image_tensors.pt", "app/utils/data/label_tensors.pt"]
    cnn = CNN(tensors=tensor_paths, model_path='app/utils/data/model_11_15_dw.pth')

    # Predict on a sample tensor from loaded image tensors

    sample_image = "/Users/kjams/Desktop/research/health_informatics/app/data/testing_data/early/WBC-Malignant-Early-010.jpg"
    prediction = cnn.predict_image(sample_image)
    print("Predicted label index:", prediction)
