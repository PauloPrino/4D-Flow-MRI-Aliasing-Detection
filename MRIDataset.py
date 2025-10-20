from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt
import torch
from PIL import Image
import torchvision.transforms.v2 as transforms

class MRIDataset(Dataset):
    def __init__(self, volumes_3D_dir, mask_dir, data_augmentation): # transform : preprocessing transformations (data augmentation)
        self.volumes_3D_dir = volumes_3D_dir
        self.mask_dir = mask_dir
        self.data_augmentation = data_augmentation
        self.volumes_3D_filenames = os.listdir(volumes_3D_dir)

    def __len__(self): # we redefine the len function to be the nbre of images in the dataset
        return len(self.volumes_3D_filenames)

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
        volume_3D_path = os.path.join(self.volumes_3D_dir, self.volumes_3D_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.volumes_3D_filenames[idx]) # we assume that the masks have the same filenames as the images
        
        volume_3D = plt.imread(volume_3D_path) # loading the image as a numpy array
        mask = plt.imread(mask_path)

        # Normalize and convert to uint8
        if volume_3D.dtype == 'float32' or volume_3D.dtype == 'float64':
            volume_3D = (255 * volume_3D).astype('uint8')
        if mask.dtype == 'float32' or mask.dtype == 'float64':
            mask = (255 * mask).astype('uint8')

        volume_3D = Image.fromarray(volume_3D) # transforming the image to PIL
        mask = Image.fromarray(mask)

        if self.data_augmentation: # if there is some data augmentation
            volume_3D, mask = self.transform_both(volume_3D, mask) # apply the transformations on both the image and corresponding mask

        volume_3D = self.transform_image(volume_3D)
        mask = self.transform_mask(mask)

        #mask = mask.convert("L") # we ensure that mask have only one channel (grayscale)

        mask = (mask > 0.5).float() # ensure mask is binary

        return volume_3D, mask

