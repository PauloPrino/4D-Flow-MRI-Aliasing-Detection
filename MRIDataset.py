from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt
import torch
from PIL import Image
import torchvision.transforms.v2 as transforms

class MRIDataset(Dataset):
    def __init__(self, image_dir, mask_dir, data_augmentation): # transform : preprocessing transformations (data augmentation)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.data_augmentation = data_augmentation
        self.image_filenames = os.listdir(image_dir)

    def __len__(self): # we redefine the len function to be the nbre of images in the dataset
        return len(self.image_filenames)

    def transform_both(self, image, mask):
        return transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ])(image, mask)

    def transform_mask(self, mask):
        return transforms.Compose([
            transforms.Resize((256, 256)), # resizing to 256x256 to avoid using too much memory but still keep enough detail
            transforms.ToTensor(),
        ])(mask)

    def transform_image(self, image):
        return transforms.Compose([
            transforms.ColorJitter(contrast=0.5),
            transforms.Resize((256, 256)), # resizing to 256x256 to avoid using too much memory but still keep enough detail
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), # normalizing
        ])(image)
    
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

        if self.data_augmentation: # if there is some data augmentation
            image, mask = self.transform_both(image, mask) # apply the transformations on both the image and corresponding mask

        image = self.transform_image(image)
        mask = self.transform_mask(mask)

        #mask = mask.convert("L") # we ensure that mask have only one channel (grayscale)

        mask = (mask > 0.5).float() # ensure mask is binary

        return image, mask

