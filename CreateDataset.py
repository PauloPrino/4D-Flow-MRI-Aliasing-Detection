import StudyCaseNIfTI
import os
import random
import numpy as np

class CreateDataset():
    def __init__(self, input_data_folder, output_data_folder, aliased_ratio, venc):
        """
        input_data_folder: the folder in which there are is all the data we'll work on, it is in the format of one subfolder per patient
        output_data_folder: the folder in which we'll output all of our data (with and without aliasing simulation on it)
        aliased_ratio: ratio of aliased data we want amongst all of the data, so it gives us how many aliasing simulations we need to perform
        """
        self.input_data_folder = input_data_folder
        self.output_data_folder = output_data_folder
        self.aliased_ratio = aliased_ratio
        self.venc = venc
        self.volumes_3D_folder = self.output_data_folder + "/3DVolumes" # the folder in which we put the 3D volumes, one direction flow by one direction flow and one time frame by one time frame (not the masks)
        self.masks_folder = self.output_data_folder + "/Masks" # the folder in which we put the masks
        self.clean_input_data_folder()
        self.create_directories()
        self.create_aliased_data(self.aliased_ratio, self.venc)
        self.move_nii_gz_files_to_output_data_folder()

    def create_directories(self):
        if not os.path.exists(self.volumes_3D_folder):
            os.makedirs(self.volumes_3D_folder)

        if not os.path.exists(self.masks_folder):
            os.makedirs(self.masks_folder)

    def clean_input_data_folder(self):
        """
        Cleans the input_data_folder by getting rid of the .json files and the magnitude nifti files
        """
        study_cases = os.listdir(self.input_data_folder)
        for study_case in study_cases:
            if study_case.endswith("_NIfTI"): # only on the NIfTIs
                for file in study_case:
                    if file.endswith(".json") or file.endswith("_NIfTI.nii.gz"): # we remove the .json file and the magnitude file
                        os.remove(file)
    
    def move_nii_gz_files_to_output_data_folder(self):
        """
        Moves the .nii.gz files that still remain in the input_data_folder after the aliasing simulation (so the ones that were not aliased) and put it in the output_data_folder in the subfolder NIfTIs time frame by time frame and flow direction by flow direction
        """
        study_cases = os.listdir(self.input_data_folder) # gives the list of all the folders of all the patients that remain after the aliasing
        for study_case in study_cases: # for each patient (each study case)
            if study_case.endswith("_NIfTI"): # only on the NIfTIs
                patient = StudyCaseNIfTI.StudyCaseNIfTI(study_case)
                flow_directions = ["LR", "AP", "SI"] # the three flow directions
                for flow_direction in flow_directions:
                    for t in range(50):
                        saving_path = self.volumes_3D_folder + "/" + study_case + "_" + flow_direction + "_t" + str(t) + ".npy" # a path like this: CleanData/3DVolumes/IRM_BAO_069_1_4D_NIfTI_LR_t5.npy
                        volume_3D = patient.get_3D_volume(flow_direction)
                        mask = np.zeros(256,256,144) # we don't put any aliasing on this so the ground truth mask is only zeros
                        np.save(saving_path, mask)

    def create_aliased_data(self, aliased_ratio, venc):
        """
        Aliases patients (entire .nii.gz file) amongst all of the data files with a ratio of aliased_ratio
        venc: the new velocity that is smaller than the one of the acquisition so that there are aliased voxels
        flow_direction: the direction of the blood flow we apply it to (LR, AP or SI)
        """
        dataset = os.listdir(self.input_data_folder) # all the files in the folder input_data_folder
        dataset_without_dicoms = []
        for data in dataset:
            if data.endswith("_NIfTI"): # only on the NIfTIs
                dataset_without_dicoms.append(data)
        total_number_of_data = len(dataset_without_dicoms) # the total number of data (the number of patients)
        nbre_aliased_data = int(np.floor(aliased_ratio * total_number_of_data)) # the number of patients we simulate aliasing on
        patients_index_to_be_aliased = [random.randint(0, total_number_of_data - 1) for _ in range(nbre_aliased_data)] # patients that we aliase

        for index in patients_index_to_be_aliased: # for each patient on which we simulate aliasing
            patient_to_be_aliased = dataset_without_dicoms[index]
            study_case = StudyCaseNIfTI.StudyCaseNIfTI(patient_to_be_aliased)
            flow_directions = ["LR", "AP", "SI"] # the three flow directions
            for flow_direction in flow_directions:
                velocity_post_aliasing, aliased_pixels, phase_data_after_aliasing = study_case.simulate_aliasing(flow_direction, venc, None) # None: because we simulate the aliasing on all the time frames
                for t in range(50): # for each time frame
                    # Save the mask
                    saving_path = self.masks_folder + "/" + patient_to_be_aliased + "_" + flow_direction + "_t" + str(t) + ".npy" # a path like this: CleanData/Masks/IRM_BAO_069_1_4D_NIfTI_LR_t5.npy
                    np.save(saving_path, aliased_pixels[:,:,:,t]) # save the 3D numpy array as a file
                    # Save the 3D Volume
                    saving_path = self.volumes_3D_folder + "/" + patient_to_be_aliased + "_" + flow_direction + "_t" + str(t) + ".npy" # a path like this: CleanData/3DVolumes/IRM_BAO_069_1_4D_NIfTI_LR_t5.npy
                    np.save(saving_path, phase_data_after_aliasing[:,:,:,t])

create_dataset = CreateDataset("Dataset/RawData", "Dataset/CleanData", 0.5, 50)