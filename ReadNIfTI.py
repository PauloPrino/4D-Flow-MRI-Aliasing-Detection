import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import GetMetadataDICOM

class StudyCaseNIfTI():
    def __init__(self, study_path):
        """study_path: path to the folder of the study"""
        self.study_path = study_path
        self.nifti_magnitude_path = study_path + "/" + study_path.split("/")[1] + ".nii.gz" # the file path of the magnitude (so the anatomy)
        self.nifti_LR_flow_path = study_path + "/" + study_path.split("/")[1] + "_e2.nii.gz" # the file path of the LR (Left-Right so axis x) of the blood flow
        self.nifti_AP_flow_path = study_path + "/" + study_path.split("/")[1] + "_e3.nii.gz" # the file path of the AP (Anterior-Posterior so axis y) of the blood flow
        self.nifti_SI_flow_path = study_path + "/" + study_path.split("/")[1] + "_e4.nii.gz" # the file path of the LR (Superior-Inferior so axis z) of the blood flow
        self.nifti_magnitude = nib.load(self.nifti_magnitude_path) # loading the NIfTI file
        self.nifti_LR_flow = nib.load(self.nifti_LR_flow_path)
        self.nifti_AP_flow = nib.load(self.nifti_AP_flow_path)
        self.nifti_SI_flow = nib.load(self.nifti_SI_flow_path)
        self.data_magnitude = self.nifti_magnitude.get_fdata() # transform the image to a numpy array
        self.data_LR_flow = self.nifti_LR_flow.get_fdata()
        self.data_AP_flow = self.nifti_AP_flow.get_fdata()
        self.data_SI_flow = self.nifti_SI_flow.get_fdata()
    
    def correspondance_nifti_file_path_protocol(self, protocol):
        if protocol == "magnitude":
            return self.nifti_magnitude_path
        if protocol == "LR":
            return self.nifti_LR_flow_path
        if protocol == "AP":
            return self.nifti_AP_flow_path
        if protocol == "SI":
            return self.nifti_SI_flow_path

    def correspondance_nifti_file_protocol(self, protocol):
        if protocol == "magnitude":
            return self.nifti_magnitude
        if protocol == "LR":
            return self.nifti_LR_flow
        if protocol == "AP":
            return self.nifti_AP_flow
        if protocol == "SI":
            return self.nifti_SI_flow
        
    def correspondance_nifti_data_protocol(self, protocol):
        if protocol == "magnitude":
            return self.data_magnitude
        if protocol == "LR":
            return self.data_LR_flow
        if protocol == "AP":
            return self.data_AP_flow
        if protocol == "SI":
            return self.data_SI_flow

    def get_venc(self, protocol):
        study_case_path = self.study_path.remove("_NIfTI")
        dicom_study_case = GetMetadataDICOM.StudyCaseDICOM(study_case_path)
        venc = dicom_study_case.get_venc(protocol)
        print(f"Velocity encoding (VENC) = {venc}mm/s")
        return venc
    
    def get_velocity_encoding_scale(self, protocol):
        study_case_path = self.study_path.remove("_NIfTI")
        dicom_study_case = GetMetadataDICOM.StudyCaseDICOM(study_case_path)
        velocity_encoding_scale = dicom_study_case.get_velocity_encode_scale(protocol)
        print(f"Velocity encoding scale = {velocity_encoding_scale}s/mm")
        return velocity_encoding_scale

    def get_header(self, protocol):
        nifti_file = self.correspondance_nifti_file_protocol(protocol)
        hdr = nifti_file.header
        print(f"NIfTI header for the protocol {protocol}: {hdr}")
        return hdr

    def get_image_shape_by_section(self, section, protocol):
        data = self.correspondance_nifti_data_protocol(protocol)
        if section == "axial":
            image_shape_x = data.shape[0]
            image_shape_y = data.shape[2]
        elif section == "coronal":
            image_shape_x = data.shape[1]
            image_shape_y = data.shape[2]
        elif section == "sagittal":
            image_shape_x = data.shape[0]
            image_shape_y = data.shape[1]
        print(f"Image size in section {section} for the protocol {protocol}: {image_shape_x} x {image_shape_y}")
        return image_shape_x, image_shape_y
    
    def get_number_of_slices_by_section(self, section, protocol):
        data = self.correspondance_nifti_data_protocol(protocol)
        if section == "axial":
            number_of_slices = data.shape[1]
        elif section == "coronal":
            number_of_slices = data.shape[0]
        elif section == "sagittal":
            number_of_slices = data.shape[2]
        print(f"Number of slices in section {section} for the protocol {protocol}: {number_of_slices}")
        return number_of_slices
    
    def get_number_of_time_frames(self, protocol):
        data = self.correspondance_nifti_data_protocol(protocol)
        number_of_frames = data.shape[3]
        print(f"Number of time frames for the protocol {protocol}: {number_of_frames}")
        return number_of_frames

    def get_slice(self, section, slice_index, time_frame, protocol):
        data = self.correspondance_nifti_data_protocol(protocol)
        if section == "sagittal": # size of a sagittal section: 256x256 (144 slices)
            # For the sagittal view we fix x and have a plan (y,z) but in the header we have that x in the world is z in nifti so we fix z in the numpy array
            slice = data[:, :, slice_index, time_frame]
        elif section == "coronal": # size of a coronal section: 256x144 (256 slices)
            # For the coronal view we fix y and have a plan (x,z) but in the header we have that y in the world is x in nifti
            slice = data[slice_index, :, :, time_frame]
        elif section == "axial": # size of an axial section: 256x144 (256 slices)
            # For the axial view we fix z and have a plan (x,y) but in the header we have that z in the world is y in nifti
            slice = data[:, slice_index, :, time_frame]
        return slice
    
    def visualize_slices(self, axial_slice_index, coronal_slice_index, sagittal_slice_index, time_frame, protocol):
        sagittal_slice = self.get_slice("sagittal", sagittal_slice_index, time_frame, protocol)
        coronal_slice = self.get_slice("coronal", coronal_slice_index, time_frame, protocol)
        axial_slice = self.get_slice("axial", axial_slice_index, time_frame, protocol)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axial_view = axes[0].imshow(np.rot90(np.rot90(axial_slice)), cmap="gray", origin="lower")
        axes[0].set_title(f"Axial Slice for protocol {protocol} at position y={axial_slice_index}") # the pixels on the image are velocities in the perpendicular direction: y

        coronal_view = axes[1].imshow(np.rot90(np.rot90(np.rot90(coronal_slice.T))), cmap="gray", origin="lower")
        axes[1].set_title(f"Coronal Slice for protocol {protocol} at position x={coronal_slice_index}") # the pixels on the image are velocities in the perpendicular direction: x

        sagital_view = axes[2].imshow(sagittal_slice.T, cmap="gray", origin="lower")
        axes[2].set_title(f"Sagittal Slice for protocol {protocol} at position z={sagittal_slice_index}") # the pixels on the image are velocities in the perpendicular direction: z

        fig.colorbar(axial_view, ax=axes[0], label="Pixel value")
        fig.colorbar(coronal_view, ax=axes[1], label="Pixel value")
        fig.colorbar(sagital_view, ax=axes[2], label="Pixel value")
        plt.tight_layout()
        plt.show()

    def visualize_slice(self, section, slice_index, time_frame, protocol):
        slice = self.get_slice(section, slice_index, time_frame, protocol)

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

    def visualize_velocity_vectors(self, section, slice_index, time_frame, scale=2, stride=4):
        """
        Visualize a slice with velocity vectors (arrows) overlaid on top of the magnitude (anatomy)
        section: 'axial', 'coronal', or 'sagittal'
        slice_index: index of the slice in that section
        time_frame: temporal frame index
        scale: scaling factor for arrow length
        stride: skip pixels to avoid overcrowding, not draw an arrow on each pixel
        """
        mag = self.data_magnitude
        vx = self.data_LR_flow # velocity in nifty (so the (y, x, z) coordinates)
        vy = self.data_AP_flow
        vz = self.data_SI_flow

        if section == "axial":
            img = mag[:, slice_index, :, time_frame]
            u = vx[:, slice_index, :, time_frame]
            v = vy[:, slice_index, :, time_frame]
            xlabel, ylabel = "Left-Right", "Anterior-Posterior"

        elif section == "coronal":
            img = mag[slice_index, :, :, time_frame]
            u = vx[slice_index, :, :, time_frame]
            v = vz[slice_index, :, :, time_frame]
            xlabel, ylabel = "Left-Right", "Superior-Inferior"

        elif section == "sagittal":
            img = mag[:, :, slice_index, time_frame]
            u = vy[:, :, slice_index, time_frame]
            v = vz[:, :, slice_index, time_frame]
            xlabel, ylabel = "Anterior-Posterior", "Superior-Inferior"

        # Downsample to avoid too many arrows (we do some striding similar as we do on )
        u = u[::stride, ::stride]
        v = v[::stride, ::stride]

        # Generate coordinates for the arrows, also downsampled
        Y, X = np.mgrid[0:img.shape[0]:stride, 0:img.shape[1]:stride]

        fig, axes = plt.subplots(1, 3, figsize=(12,4))
        axes[0].imshow(img, cmap='gray', origin='lower')
        u_view = axes[1].imshow(u, cmap="gray", origin="lower")
        v_view = axes[2].imshow(v, cmap="gray", origin="lower")
        fig.colorbar(u_view, ax=axes[1], label="Pixel value")
        fig.colorbar(v_view, ax=axes[2], label="Pixel value")
        axes[0].quiver( # overlay arrows on top of the image
            X, Y, u, v,
            color='red',
            scale=scale,
            width=0.002,
            angles='xy',
            scale_units='xy',
            headwidth=3, # width of the head of the arrow
            headlength=4, # length of the head of the arrow
            headaxislength=3 # length of the body of the arrow
        )
        axes[0].set_title(f"{section.capitalize()} slice {slice_index} - Velocity field (t={time_frame})")
        axes[0].set_xlabel(xlabel)
        axes[0].set_ylabel(ylabel)
        plt.tight_layout()
        plt.show()


nifti_file = StudyCaseNIfTI("Dataset/IRM_BAO_069_1_4D_NIfTI")
nifti_file.get_header("LR")
nifti_file.get_image_shape_by_section("coronal","LR")
nifti_file.get_number_of_slices_by_section("coronal","LR")
nifti_file.get_number_of_time_frames("LR")
nifti_file.visualize_slices(94, 189, 68,0,"LR")
nifti_file.visualize_slice("coronal", 189, 0,"LR")
nifti_file.get_venc("LR")
nifti_file.visualize_velocity_vectors("sagittal", 73, 0, scale=50, stride=2)