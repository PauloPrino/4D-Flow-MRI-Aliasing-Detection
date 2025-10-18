import StudyCaseNIfTI
import os
import random
import numpy as np

class CreateDataset():
    def __init__(self, input_data_folder, output_data_folder, aliased_ratio):
        """
        input_data_folder: the folder in which there are is all the data we'll work on, it is in the format of one subfolder per patient
        output_data_folder: the folder in which we'll output all of our data (with and without aliasing simulation on it)
        aliased_ratio: ratio of aliased data we want amongst all of the data, so it gives us how many aliasing simulations we need to perform
        """
        self.input_data_folder = input_data_folder
        self.output_data_folder = output_data_folder
        self.aliased_ratio = aliased_ratio
        self.nifti_folder = self.output_data_folder + "NIfTI" # the folder in which we put the nifti files (not the masks)
        self.masks_folder = self.output_data_folder + "masks" # the folder in which we put the masks
        self.create_directories()

    def create_directories(self):
        if not os.path.exists(self.nifti_folder):
            os.makedirs(self.nifti_folder)

        if not os.path.exists(self.masks_folder):
            os.makedirs(self.masks_folder)

    def save_aliasing_mask(self, aliased_array, saving_path):
        """
        Saves the 3D numpy array aliasing mask as a .npy file: values 0 for non aliased voxels and values 1 for aliased voxels
        aliased_array: the numpy array of the aliased voxels
        saving_path: the path to which we save the file .npy
        """
        np.save(saving_path, aliased_array)
    
    def move_nii_gz_files_to_output_data_folder(self):
        """
        Moves the .nii.gz files in the input_data_folder and put it in the output_data_folder
        """
        study_cases = os.listdir(self.input_data_folder) # gives the list of all the folders of all the patients we study
        for study_case in study_cases: # for each patient (each study case)
            if study_case.endswith(".nii.gz"): # if it is a nifti file, not the json file associated with it
                os.rename(study_case, self.output_data_folder + study_case)

    def create_aliased_data(self, aliased_ratio, venc):
        """
        Aliases patients (entire .nii.gz file) amongst all of the data files with a ratio of aliased_ratio
        venc: the new velocity that is smaller than the one of the acquisition so that there are aliased voxels
        flow_direction: the direction of the blood flow we apply it to (LR, AP or SI)
        """
        dataset = os.listdir(self.output_data_folder) # all the files in the folder output_data_folder
        total_number_of_data = len(dataset) # the number of data (the number of patients)
        nbre_aliased_data = aliased_ratio * total_number_of_data # the number of patients we simulate aliasing on
        files_index_to_be_aliased = [random.randint(0, total_number_of_data - 1) for _ in range(nbre_aliased_data)]

        for index in range(files_index_to_be_aliased): # for each patient on which we simulate aliasing
            file_to_be_aliased = dataset[index]
            study_case = StudyCaseNIfTI.StudyCaseNIfTI(file_to_be_aliased)
            flow_directions = ["LR", "AP", "SI"] # the three flow directions
            for flow_direction in flow_directions:
                velocity_post_aliasing, aliased_pixels, phase_data_after_aliasing = study_case.simulate_aliasing(flow_direction, venc, None) # None: because we simulate the aliasing on all the time frames
                for t in range(50): # for each time frame
                    saving_path = file_to_be_aliased.replace(self.input_data_folder, "") + "t" + str(t)
                    self.save_aliasing_mask(aliased_pixels[:,:,:,t], saving_path)