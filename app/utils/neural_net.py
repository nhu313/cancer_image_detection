import os
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from utils.load_images import ImageLoad  # Assuming ImageLoad is saved in load_images.py


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
