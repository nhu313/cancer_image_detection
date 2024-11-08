import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class CNN():
    def __init__(self, architecture: str = "wide", tensors: list = None, model_path: str = "cnn_model.pth"):
        self.input_channels = 3
        self.number_of_labels = 4
        self.size = (64,64)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.build_model(architecture).to(self.device)
        
        
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
            architecture (str): Architecture type ("wide").

        Returns:
            nn.Sequential: A sequential model based on the desired architecture.
        """
        layers = []
        if architecture == "deep-wide":
    # Wide and deep architecture
            layers.extend([
        # Convolution Block 1
        nn.Conv2d(self.input_channels, 64, kernel_size=5, stride=1, padding=2),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout(0.3),

        # Convolution Block 2
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout(0.3),

        # Convolution Block 3
        nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout(0.4),

        # Convolution Block 4
        nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout(0.4),

        # Flatten and Fully Connected Layers
        nn.Flatten(),
        nn.Linear(512 * 4 * 4, 1024),  # Adjust based on image size
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        
        # Final output layer
        nn.Linear(512, self.number_of_labels),
        nn.LogSoftmax(dim=1)  # LogSoftmax for multi-class classification
    ])

        return nn.Sequential(*layers)    
   
    def load_model(self, model_path):
        # Load model parameters
        self.model.load_state_dict(torch.load(model_path, weights_only=True, map_location=self.device))
        self.model.eval()
        print("Model loaded from disk.")
        
    def predict_image(self, path_to_image) -> int:
        # Predict class for a single image tensor
        self.model.eval()
        img = self._open_img(image_path=path_to_image, add_noise=False)
        img_tensor = torch.tensor(img, dtype=torch.float32) / 255.0  # Normalize to [0, 1]
        img_tensor = F.normalize(img_tensor)

        img_tensor = img_tensor.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        img_tensor.to(self.device)  # Return tensor on correct device
        with torch.no_grad():
            output = self.model(img_tensor.unsqueeze(0))  # Add batch dimension
            predicted_label_index = torch.argmax(output, dim=1).item()
            return predicted_label_index

    def test_model(self, test_image_tensors: torch.Tensor=None, test_label_tensors: torch.Tensor = None) -> float:
        """
        Tests the CNN model on a set of test images.

        Args:
            test_image_tensors (torch.Tensor): Tensors of test images.
            test_label_tensors (torch.Tensor): Tensors of true labels for the test images.

        Returns:
            float: Test accuracy as a percentage.
        """
        self.model.eval()  # Set the model to evaluation mode
        correct = 0  # Counter for correct predictions
        total = len(self.label_tensors) # Total number of test samples
        print('Total:',total)
        with torch.no_grad():  # Disable gradient computation for testing
            for i in tqdm(range(total)):
                img_tensor = self.image_tensors[i].unsqueeze(0).to(self.device)  # Add batch dimension
                label = self.label_tensors[i].to(self.device)
                output = self.model(img_tensor)  # Forward pass

                # Get the predicted label (index with max probability)
                predicted_label_index = torch.argmax(output, dim=1).item()
                
                # Check if the prediction matches the true label
                if predicted_label_index == label.item():
                    correct += 1
        
            accuracy = (correct / total) * 100
            print(f"Test Accuracy: {accuracy:.2f}%")
        return accuracy

# Example usage
if __name__ == "__main__":
    tensor_paths = ["app/utils/data/image_tensors.pt", "app/utils/data/label_tensors.pt"]
    cnn = CNN(tensors=tensor_paths, model_path='app/utils/data/model_11_4.pth')

    # Predict on a sample tensor from loaded image tensors

    sample_image = "/Users/kjams/Desktop/research/health_informatics/app/data/testing_data/early/WBC-Malignant-Early-010.jpg"
    prediction = cnn.predict_image(sample_image)
    print("Predicted label index:", prediction)
