import os
import cv2
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import random
import pandas as pd


class ImageLoad:
    def __init__(self, dataset_path: str, save_new_dataset: bool = 0, resize_to=(64, 64)):
        """
        Initializes the ImageLoad class.

        Args:
            dataset_path (str): Path to the dataset directory containing images.
            resize_to (tuple): Dimensions to resize images to (default: (64, 64)).
        """
        self.save_new_dataset = save_new_dataset
        self.size = resize_to
        self.dataset_path = dataset_path
        self.image_np_arrays = list() # populated in main_loop
        self.output_dir = 'data/output'
        os.makedirs(self.output_dir, exist_ok=True)

        # Load image paths
        self.image_paths = self._load_image_paths()
        # dataframe is populated in main_loop
        self.df = pd.DataFrame()
        self.main_loop()
    

    def main_loop(self) -> dict:
        """
        Processes all images and stores the results.
        """
        #all_together_arrays = []
        print('Start batch process...')

        for idx, (fname, img_path, category) in enumerate(tqdm(self.image_paths, desc="Processing images")):
            image = self._open_img(img_path)

            processed_images = [
                image,
                self._add_gaussian_blurr(image),
                self._rotate_180(image),
                self._rotate_90_clockwise(image),
                self._rotate_90_counter_clockwise(image)
            ]
            self.image_np_arrays.append((category, processed_images))

            if self.save_new_dataset == 1:
                for idx_i, img in enumerate(processed_images):
                    id_ = f"{idx + 1}_{idx_i}"
                    self.save_to_folders(fname, img, id_, category)
        

        # Parse each img into df as cols: (Image, Numpy Array)
        self.df = pd.DataFrame(
            [(cat, image) for cat, imgs in self.image_np_arrays for image in imgs],
                columns=['Label', 'Image']
        )
        print('Batch process completed.')
        #return all_together_arrays


    def _open_img(self, image_path: str, add_noise:bool= True) -> np.ndarray:
        """Opens and transforms the image into NumPy array form."""
        img = cv2.imread(image_path)
        img_resized = cv2.resize(img, self.size)  # Resize the image
        # TODO add noise to each image
        if add_noise and random.randint(0, 1) > 0.7:
            return self._add_gaussian_noise(img_resized)
        return img_resized

    def _add_gaussian_noise(self, image: np.ndarray, mean: float = 0, sigma: float = 25) -> np.ndarray:
        """Adds Gaussian noise to the image."""

        noise = np.random.normal(mean, sigma, image.shape).astype(np.uint8)
        # Add the noise to the image
        return cv2.add(image, noise)

    def save_to_folders(self, fname: str, image: np.ndarray, idx: str, category: str):
        """Saves the processed image to the specified category folder."""
        # Ensure image is in the correct format
        if not isinstance(image, np.ndarray):
            raise ValueError("The image must be a NumPy array.")

        # Create the category folder if it doesn't exist
        category_path = os.path.join(self.output_dir, 'processed', category)
        os.makedirs(category_path, exist_ok=True)

        # Construct the full file path
        file_path = os.path.join(category_path, f"{idx}_{fname}")
        print(f"Saving image to: {file_path}")

        # Save the image
        success = cv2.imwrite(file_path, image)
        if not success:
            raise IOError(f"Failed to save the image at {file_path}")

    def _rotate_90_counter_clockwise(self, image: np.ndarray) -> np.ndarray:
        """Rotates the image 90 degrees counterclockwise."""
        return np.rot90(image, k=1)

    def _rotate_90_clockwise(self, image: np.ndarray) -> np.ndarray:
        """Rotates the image 90 degrees clockwise."""
        return np.rot90(image, k=-1)

    def _rotate_180(self, image: np.ndarray) -> np.ndarray:
        """Rotates the image 180 degrees."""
        return cv2.flip(image, -1)

    def _add_gaussian_blurr(self, image: np.ndarray) -> np.ndarray:
        """Applies Gaussian blur to the image."""
        return cv2.GaussianBlur(image, (7, 7), 0)
    
    def _load_image_paths(self) -> list[tuple[str, str, str]]:
        """
        Recursively loads image file paths from the dataset directory.

        Returns:
            list[tuple[str, str, str]]: List of image file paths with their respective category.
        """
        image_extensions = ['.tiff', '.tif', '.jpg', '.jpeg', '.png']
        image_paths = []

        # Loop through the main dataset directory
        for category in os.listdir(self.dataset_path):
            category_path = os.path.join(self.dataset_path, category)
            if os.path.isdir(category_path):
                for fname in os.listdir(category_path):
                    if any(fname.lower().endswith(ext) for ext in image_extensions):
                        image_paths.append(
                            (fname, os.path.join(category_path, fname), category))

        return image_paths
    
   
        """
        Loads an image from the given file path and converts it to a tensor.

        Args:
            file_path (str): Path to the image file.

        Returns:
            torch.Tensor: The transformed image tensor of shape (3, 64, 64).
        """
        # Load the image
        image = Image.open(file_path).convert('RGB')  # Convert to RGB if needed

        # Define a transformation to resize and convert to tensor
        transform = transforms.Compose([
            transforms.Resize((64, 64)),  # Resize to 64x64
            transforms.ToTensor(),  # Convert to tensor
        ])

        # Apply the transformation
        image_tensor = transform(image)
        return image_tensor