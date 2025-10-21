from torch.utils.data import Dataset
import os
import torch
import numpy as np

class MRIDataset(Dataset):
    def __init__(self, volumes_3D_dir, mask_dir): # transform : preprocessing transformations (data augmentation)
        self.volumes_3D_dir = volumes_3D_dir
        self.mask_dir = mask_dir
        self.volumes_3D_filenames = os.listdir(volumes_3D_dir)

    def __len__(self): # we redefine the len function to be the nbre of images in the dataset
        return len(self.volumes_3D_filenames)
    
    def __getitem__(self, idx):
        volume_3D_path = os.path.join(self.volumes_3D_dir, self.volumes_3D_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.volumes_3D_filenames[idx]) # we assume that the masks have the same filenames as the images
        
        volume_3D = np.load(volume_3D_path).astype(np.float32) # loading the 3D volume that is saved as a numpy array and converting it to float32 as PyTorch uses floats and not int
        mask = np.load(mask_path).astype(np.float32)

        volume_3D = (volume_3D - volume_3D.min()) / (volume_3D.max() - volume_3D.min()) # normalize the values between 0 and 1: [0,1]

        volume_3D = torch.from_numpy(volume_3D).unsqueeze(0) # convert to tensor
        mask = torch.from_numpy(mask).unsqueeze(0) # we use unsqueeze(0) because the numpy array is of size (depth, height, width) but the Conv3d in UNET expects (1(number of in channels), depth, height, width)

        #mask = mask.convert("L") # we ensure that mask have only one channel (grayscale)

        #mask = (mask > 0.5).float() # ensure mask is binary

        return volume_3D, mask