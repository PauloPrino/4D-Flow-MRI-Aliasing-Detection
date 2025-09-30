from torch.utils.data import Dataset
import os

class MRIDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform): # transform : preprocessing transformations (data augmentation)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_filenames = os.listdir(image_dir)

    def __len__(self): # we redefine the len function to be the nbre of images in the dataset
        return len(self.image_filenames)

