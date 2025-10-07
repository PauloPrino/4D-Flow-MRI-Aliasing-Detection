from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt
import torch
from PIL import Image

class MRIDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_transform, mask_transform): # transform : preprocessing transformations (data augmentation)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.image_filenames = os.listdir(image_dir)

    def __len__(self): # we redefine the len function to be the nbre of images in the dataset
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.image_filenames[idx]) # we assume that the masks have the same filenames as the images
        
        image = plt.imread(img_path) # loading the image as a numpy array
        mask = plt.imread(mask_path)

        # Normalize and convert to uint8
        if image.dtype == 'float32' or image.dtype == 'float64':
            image = (255 * image).astype('uint8')
        if mask.dtype == 'float32' or mask.dtype == 'float64':
            mask = (255 * mask).astype('uint8')

        image = Image.fromarray(image) # transforming the image to PIL
        mask = Image.fromarray(mask)

        if self.image_transform: # if there is some preprocessing on the data
            image = self.image_transform(image) # apply the transformations on both the image and corresponding mask
        mask = mask.convert("L") # we ensure that mask have only one channel (grayscale)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        mask = (mask > 0.5).float() # ensure mask is binary

        return image, mask

