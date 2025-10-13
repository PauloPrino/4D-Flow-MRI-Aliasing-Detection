import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

class ReadNIfTI():
    def __init__(self, nii_gz_path):
        """nii_gz: path of the .nii.gz file"""
        self.nii_gz_path = nii_gz_path
        self.nifti_file = nib.load(nii_gz_path) # loading the NIfTI file
        self.data = self.nifti_file.get_fdata() # transform the image to a numpy array
    
    def get_header(self):
        hdr = self.nifti_file.header
        print(f"NIfTI header: {hdr}")
        return hdr

    def get_image_shape_by_section(self, section):
        if section == "axial":
            image_shape_x = self.data.shape[0]
            image_shape_y = self.data.shape[2]
        elif section == "coronal":
            image_shape_x = self.data.shape[1]
            image_shape_y = self.data.shape[2]
        elif section == "sagittal":
            image_shape_x = self.data.shape[0]
            image_shape_y = self.data.shape[1]
        print(f"Image size in section {section}: {image_shape_x} x {image_shape_y}")
        return image_shape_x, image_shape_y
    
    def get_number_of_slices_by_section(self, section):
        if section == "axial":
            number_of_slices = self.data.shape[1]
        elif section == "coronal":
            number_of_slices = self.data.shape[0]
        elif section == "sagittal":
            number_of_slices = self.data.shape[2]
        print(f"Number of slices in section {section}: {number_of_slices}")
        return number_of_slices
    
    def get_number_of_time_frames(self):
        number_of_frames = self.data.shape[3]
        print(f"Number of time frames: {number_of_frames}")
        return number_of_frames

    def get_slice(self, section, slice_index, time_frame):
        if section == "sagittal": # size of a sagittal section: 256x256 (144 slices)
            slice = self.data[:, :, slice_index, time_frame] # image in the plan (x,y)
        elif section == "coronal": # size of a coronal section: 256x144 (256 slices)
            slice = self.data[slice_index, :, :, time_frame] # image in the plan (y,z)
        elif section == "axial": # size of an axial section: 256x144 (256 slices)
            slice = self.data[:, slice_index, :, time_frame] # image in the plan (x,z)
        return slice
    
    def visualize_slices(self, axial_slice_index, coronal_slice_index, sagittal_slice_index, time_frame):
        sagittal_slice = self.get_slice("sagittal", sagittal_slice_index, time_frame)
        coronal_slice = self.get_slice("coronal", coronal_slice_index, time_frame)
        axial_slice = self.get_slice("axial", axial_slice_index, time_frame)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axial_view = axes[0].imshow(np.rot90(np.rot90(axial_slice)), cmap="gray", origin="lower")
        axes[0].set_title(f"Axial Slice at position y={axial_slice_index}") # the pixels on the image are velocities in the perpendicular direction: y

        coronal_view = axes[1].imshow(np.rot90(np.rot90(np.rot90(coronal_slice.T))), cmap="gray", origin="lower")
        axes[1].set_title(f"Coronal Slice at position x={coronal_slice_index}") # the pixels on the image are velocities in the perpendicular direction: x

        sagital_view = axes[2].imshow(sagittal_slice.T, cmap="gray", origin="lower")
        axes[2].set_title(f"Sagittal Slice at position z={sagittal_slice_index}") # the pixels on the image are velocities in the perpendicular direction: z

        fig.colorbar(axial_view, ax=axes[0], label="Pixel value")
        fig.colorbar(coronal_view, ax=axes[1], label="Pixel value")
        fig.colorbar(sagital_view, ax=axes[2], label="Pixel value")
        plt.tight_layout()
        plt.show()

    def visualize_slice(self, section, slice_index, time_frame):
        slice = self.get_slice(section, slice_index, time_frame)

        fig, axes = plt.subplots(1, 1, figsize=(12, 4))
        if section == "axial":
            view = axes.imshow(np.rot90(np.rot90(slice)), cmap="gray", origin="lower")
        elif section == "coronal":
            view = axes.imshow(np.rot90(np.rot90(np.rot90(slice.T))), cmap="gray", origin="lower")
        elif section == "sagittal":
            view = axes.imshow(slice.T, cmap="gray", origin="lower")
        axes.set_title(f"{section} Slice at position ={slice_index}") # the pixels on the image are velocities in the direction perpendicular to the section

        fig.colorbar(view, ax=axes, label="Pixel value")

        plt.tight_layout()
        plt.show()

nifti_file = ReadNIfTI("Dataset/IRM_BAO_069_1_4D_NIfTI/IRM_BAO_069_1_4D_NIfTI.nii.gz")
nifti_file.get_header()
nifti_file.get_image_shape_by_section("coronal")
nifti_file.get_number_of_slices_by_section("coronal")
nifti_file.get_number_of_time_frames()
nifti_file.visualize_slices(94, 189, 68,0)
nifti_file.visualize_slice("coronal", 189, 0)