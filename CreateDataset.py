import StudyCaseNIfTI
import os
import random
import numpy as np
import time

class CreateDataset():
    def __init__(self, input_data_folder, output_data_folder, aliased_ratio, venc):
        """
        input_data_folder: the folder in which there are is all the data we'll work on, it is in the format of one subfolder per patient
        output_data_folder: the folder in which we'll output all of our data (with and without aliasing simulation on it)
        aliased_ratio: ratio of aliased data we want amongst all of the data, so it gives us how many aliasing simulations we need to perform
        """
        start_time = time.time()
        self.input_data_folder = input_data_folder
        self.output_data_folder = output_data_folder
        self.aliased_ratio = aliased_ratio
        self.venc = venc
        self.volumes_3D_folder = self.output_data_folder + "/3DVolumes" # the folder in which we put the 3D volumes, one direction flow by one direction flow and one time frame by one time frame (not the masks)
        self.masks_folder = self.output_data_folder + "/Masks" # the folder in which we put the masks
        self.patients_to_be_aliased = []
        self.patients_not_to_be_aliased = []
        self.create_directories()
        self.distribution_of_patients_between_to_be_and_not_to_be_aliased()
        self.create_aliased_data()
        self.move_nii_gz_files_to_output_data_folder()
        print(f"Time to create the dataset: {time.time() - start_time}s")

    def create_directories(self):
        if not os.path.exists(self.volumes_3D_folder):
            os.makedirs(self.volumes_3D_folder)

        if not os.path.exists(self.masks_folder):
            os.makedirs(self.masks_folder)
    
    def distribution_of_patients_between_to_be_and_not_to_be_aliased(self):
        """
        Defines the patients that have to be aliased and the ones that do not
        """
        dataset = os.listdir(self.input_data_folder) # all the patients in the folder input_data_folder
        total_number_of_data = len(dataset) # the total number of data (the number of patients)
        nbre_aliased_data = int(np.floor(self.aliased_ratio * total_number_of_data)) # the number of patients we simulate aliasing on
        patients_index_to_be_aliased = [random.randint(0, total_number_of_data - 1) for _ in range(nbre_aliased_data)] # patients that we aliase
        for index in patients_index_to_be_aliased: # for each patient on which we simulate aliasing
            self.patients_to_be_aliased.append(dataset[index])

        for index in range(total_number_of_data):
            if index not in patients_index_to_be_aliased: # if this index is not to be aliased
                self.patients_not_to_be_aliased.append(dataset[index])

    def move_nii_gz_files_to_output_data_folder(self):
        """
        Moves the .nii.gz files that still remain in the input_data_folder after the aliasing simulation (so the ones that were not aliased) and put it in the output_data_folder in the subfolder NIfTIs time frame by time frame and flow direction by flow direction
        """
        print("Moving the remaining patients that are not being aliased...")

        for study_case in self.patients_not_to_be_aliased: # for each patient that we dont want to aliase (each study case)
            if study_case.endswith("_NIfTI"): # only on the NIfTIs
                patient = StudyCaseNIfTI.StudyCaseNIfTI(os.path.join(self.input_data_folder, study_case))
                flow_directions = ["LR", "AP", "SI"] # the three flow directions
                for flow_direction in flow_directions:
                    for t in range(50):
                        saving_path = self.volumes_3D_folder + "/" + study_case + "_" + flow_direction + "_t" + str(t) + ".npy" # a path like this: CleanData/3DVolumes/IRM_BAO_069_1_4D_NIfTI_LR_t5.npy
                        volume_3D = patient.get_3D_volume(t, flow_direction)
                        np.save(saving_path, volume_3D)
                        saving_path = self.masks_folder + "/" + study_case + "_" + flow_direction + "_t" + str(t) + ".npy" # a path like this: CleanData/Masks/IRM_BAO_069_1_4D_NIfTI_LR_t5.npy
                        mask = np.zeros((256,256,144), dtype=bool) # we don't put any aliasing on this so the ground truth mask is only zeros
                        np.save(saving_path, mask)

    def create_aliased_data(self):
        """
        Aliases patients (entire .nii.gz file) amongst all of the data files with a ratio of aliased_ratio
        venc: the new velocity that is smaller than the one of the acquisition so that there are aliased voxels
        flow_direction: the direction of the blood flow we apply it to (LR, AP or SI)
        """
        print("Simulating aliasing...")
        for patient_to_be_aliased in self.patients_to_be_aliased:
            print(f"Simulating aliasing on patient {os.path.join(self.input_data_folder, patient_to_be_aliased)}")
            study_case = StudyCaseNIfTI.StudyCaseNIfTI(os.path.join(self.input_data_folder, patient_to_be_aliased))
            flow_directions = ["LR", "AP", "SI"] # the three flow directions
            for flow_direction in flow_directions:
                velocity_post_aliasing, aliased_pixels, phase_data_after_aliasing = study_case.simulate_aliasing(flow_direction, self.venc, None) # None: because we simulate the aliasing on all the time frames
                for t in range(50): # for each time frame
                    # Save the mask
                    saving_path = self.masks_folder + "/" + patient_to_be_aliased + "_" + flow_direction + "_t" + str(t) + ".npy" # a path like this: CleanData/Masks/IRM_BAO_069_1_4D_NIfTI_LR_t5.npy
                    np.save(saving_path, aliased_pixels[:,:,:,t]) # save the 3D numpy array as a file
                    # Save the 3D Volume
                    saving_path = self.volumes_3D_folder + "/" + patient_to_be_aliased + "_" + flow_direction + "_t" + str(t) + ".npy" # a path like this: CleanData/3DVolumes/IRM_BAO_069_1_4D_NIfTI_LR_t5.npy
                    np.save(saving_path, phase_data_after_aliasing[:,:,:,t])

create_dataset = CreateDataset("Dataset/RawData", "Dataset/CleanData", 1, 50)