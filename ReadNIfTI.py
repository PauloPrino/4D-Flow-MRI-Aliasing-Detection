import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import GetMetadataDICOM

MAX_PIXEL_VALUE = 32768 # encoded in int16

class StudyCaseNIfTI():
    def __init__(self, study_path):
        """
        study_path: path to the folder of the study
        """
        self.study_path = study_path
        self.nifti_magnitude_path = study_path + "/" + study_path.split("/")[1] + ".nii.gz" # the file path of the magnitude (so the anatomy)
        self.nifti_LR_flow_path = study_path + "/" + study_path.split("/")[1] + "_e2.nii.gz" # the file path of the LR (Left-Right so axis x) of the blood flow
        self.nifti_AP_flow_path = study_path + "/" + study_path.split("/")[1] + "_e3.nii.gz" # the file path of the AP (Anterior-Posterior so axis y) of the blood flow
        self.nifti_SI_flow_path = study_path + "/" + study_path.split("/")[1] + "_e4.nii.gz" # the file path of the LR (Superior-Inferior so axis z) of the blood flow
        self.nifti_magnitude = nib.load(self.nifti_magnitude_path) # loading the NIfTI file
        self.nifti_LR_flow = nib.load(self.nifti_LR_flow_path)
        self.nifti_AP_flow = nib.load(self.nifti_AP_flow_path)
        self.nifti_SI_flow = nib.load(self.nifti_SI_flow_path)
        self.data_magnitude = self.nifti_magnitude.get_fdata(dtype=np.float32) # transform the image to a numpy array
        self.data_LR_flow = self.nifti_LR_flow.get_fdata(dtype=np.float32) # dtype float32 to not take too much memory space (4 bytes only as int16 is only 2 bytes)
        self.data_AP_flow = self.nifti_AP_flow.get_fdata(dtype=np.float32)
        self.data_SI_flow = self.nifti_SI_flow.get_fdata(dtype=np.float32)
    
    def correspondance_nifti_file_path_protocol(self, protocol):
        """
        INPUT:
            - protocol: the protocol chosen amongst magnitude(anatomy), LR(Left-Right flow), AP(Anterior-Posterior flow), SI(Superior-Inferior flow)

        OUTPUT: the path of the .nii.gz file corresponding to the chosen protocol
        """
        if protocol == "magnitude":
            return self.nifti_magnitude_path
        if protocol == "LR":
            return self.nifti_LR_flow_path
        if protocol == "AP":
            return self.nifti_AP_flow_path
        if protocol == "SI":
            return self.nifti_SI_flow_path

    def correspondance_nifti_file_protocol(self, protocol):
        """
        INPUT:
            - protocol: the protocol chosen amongst magnitude(anatomy), LR(Left-Right flow), AP(Anterior-Posterior flow), SI(Superior-Inferior flow)

        OUTPUT: the loaded .nii.gz file corresponding to the chosen protocol
        """
        if protocol == "magnitude":
            return self.nifti_magnitude
        if protocol == "LR":
            return self.nifti_LR_flow
        if protocol == "AP":
            return self.nifti_AP_flow
        if protocol == "SI":
            return self.nifti_SI_flow
        
    def correspondance_nifti_data_protocol(self, protocol):
        """
        INPUT:
            - protocol: the protocol chosen amongst magnitude(anatomy), LR(Left-Right flow), AP(Anterior-Posterior flow), SI(Superior-Inferior flow)

        OUTPUT: the data (stored in numpy format) of the file corresponding to the chosen protocol
        """
        if protocol == "magnitude":
            return self.data_magnitude
        if protocol == "LR":
            return self.data_LR_flow
        if protocol == "AP":
            return self.data_AP_flow
        if protocol == "SI":
            return self.data_SI_flow

    def get_venc(self, protocol):
        """
        INPUT:
            - protocol: the protocol chosen amongst magnitude(anatomy), LR(Left-Right flow), AP(Anterior-Posterior flow), SI(Superior-Inferior flow)

        OUTPUT: 
            - venc: the velocity encoding of the protocol chosen
        """
        study_case_path = self.study_path.replace("_NIfTI","")
        dicom_study_case = GetMetadataDICOM.StudyCaseDICOM(study_case_path)
        venc = dicom_study_case.get_venc(protocol)
        return venc
    
    def get_venc_scale(self, protocol):
        """
        INPUT:
            - protocol: the protocol chosen amongst magnitude(anatomy), LR(Left-Right flow), AP(Anterior-Posterior flow), SI(Superior-Inferior flow)
        
        OUTPUT:
            - venc_scale: the velocity encoding scale
        """
        study_case_path = self.study_path.replace("_NIfTI","")
        dicom_study_case = GetMetadataDICOM.StudyCaseDICOM(study_case_path)
        venc_scale = dicom_study_case.get_velocity_encode_scale(protocol)
        print(f"Velocity encoding scale = {venc_scale}s/mm")
        return venc_scale

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
    
    def get_slice_velocity(self, time_frame, section, protocol, slice_index):
        venc = self.get_venc(protocol)
        protocol_data = self.correspondance_nifti_data_protocol(protocol)
        velocity_data = (protocol_data[:, :, :, time_frame] / MAX_PIXEL_VALUE) * venc # in cm/s
        if section == "sagittal":
            return velocity_data[:, :, slice_index]
        if section == "coronal":
            return velocity_data[slice_index, :, :]
        if section == "axial":
            return velocity_data[:, slice_index, :]
    
    def visualize_slices(self, axial_slice_index, coronal_slice_index, sagittal_slice_index, time_frame, protocol):
        sagittal_slice = self.get_slice("sagittal", sagittal_slice_index, time_frame, protocol)
        coronal_slice = self.get_slice("coronal", coronal_slice_index, time_frame, protocol)
        axial_slice = self.get_slice("axial", axial_slice_index, time_frame, protocol)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axial_view = axes[0].imshow(np.rot90(np.rot90(axial_slice)), cmap="gray", origin="lower")
        axes[0].set_title(f"Axial Slice for protocol {protocol} at position y={axial_slice_index}") # the pixels on the image are velocities in the perpendicular direction: y

        coronal_view = axes[1].imshow(np.rot90(np.rot90(np.rot90(coronal_slice.T))), cmap="gray", origin="lower")
        axes[1].set_title(f"Coronal Slice for protocol {protocol} at position x={coronal_slice_index}") # the pixels on the image are velocities in the perpendicular direction: x

        sagittal_view = axes[2].imshow(sagittal_slice.T, cmap="gray", origin="lower")
        axes[2].set_title(f"Sagittal Slice for protocol {protocol} at position z={sagittal_slice_index}") # the pixels on the image are velocities in the perpendicular direction: z

        fig.colorbar(axial_view, ax=axes[0], label="Phase value")
        fig.colorbar(coronal_view, ax=axes[1], label="Phase value")
        fig.colorbar(sagittal_view, ax=axes[2], label="Phase value")
        plt.tight_layout()
        plt.show()

        return axial_view, coronal_view, sagittal_view

    def visualize_slice(self, section, slice_index, time_frame, protocol):
        """
        Visualize a slice with the phase values
        """
        slice = self.get_slice(section, slice_index, time_frame, protocol)

        fig, axes = plt.subplots(1, 1, figsize=(12, 4))
        if section == "axial":
            view = axes.imshow(np.rot90(np.rot90(slice)), cmap="gray", origin="lower")
        elif section == "coronal":
            view = axes.imshow(np.rot90(np.rot90(np.rot90(slice.T))), cmap="gray", origin="lower")
        elif section == "sagittal":
            view = axes.imshow(slice.T, cmap="gray", origin="lower")
        axes.set_title(f"{section} Slice at position ={slice_index}") # the pixels on the image are velocities in the direction perpendicular to the section

        fig.colorbar(view, ax=axes, label="Phase value")

        plt.tight_layout()
        plt.show()

    def visualize_velocity_slice(self, section, slice_index, time_frame, protocol):
        """
        Visualize a slice with its velocity values and not phase values
        """
        slice_velocity = self.get_slice_velocity(time_frame, section, protocol, slice_index)      

        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(slice_velocity.T, cmap="seismic", origin="lower")
        ax.set_title(f"{section.capitalize()} slice {slice_index} - Velocity (cm/s)")
        cbar = fig.colorbar(im, ax=ax, label="Velocity (cm/s)")
        plt.show()

    def conversion_phase_to_velocity(self, protocol):
        """
        Converts phase data (direct data comming from the MRI machine) to velocity data using the venc value
        """
        venc = self.get_venc(protocol)
        protocol_data = self.correspondance_nifti_data_protocol(protocol)
        print(f"Converting phase pixel values into velocities using VENC={venc}cm/s")
        velocities = (protocol_data / MAX_PIXEL_VALUE) * venc
        return velocities

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
        vx = self.conversion_phase_to_velocity("LR") # velocity in nifti (so the (y, x, z) coordinates)
        vy = self.conversion_phase_to_velocity("AP")
        vz = self.conversion_phase_to_velocity("SI")

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

        vector_norm = np.sqrt(u**2 + v**2) # norm of the velocity

        # Generate coordinates for the arrows, also downsampled
        Y, X = np.mgrid[0:img.shape[0]:stride, 0:img.shape[1]:stride]

        fig, axes = plt.subplots(1, 3, figsize=(12,4))
        axes[0].imshow(img, cmap='gray', origin='lower')
        u_view = axes[1].imshow(u, cmap="gray", origin="lower")
        v_view = axes[2].imshow(v, cmap="gray", origin="lower")
        fig.colorbar(u_view, ax=axes[1], label="Phase value")
        fig.colorbar(v_view, ax=axes[2], label="Phase value")
        norm = plt.Normalize(vmin=np.min(vector_norm), vmax=np.max(vector_norm))
        cmap = plt.colormaps['plasma']
        vector_view = axes[0].quiver( # overlay arrows on top of the image
            X, Y, u, v,
            vector_norm,
            cmap=cmap,
            norm=norm,
            scale=scale,
            width=0.002,
            angles='xy',
            scale_units='xy',
            headwidth=3, # width of the head of the arrow
            headlength=4, # length of the head of the arrow
            headaxislength=3 # length of the body of the arrow
        )
        fig.colorbar(vector_view, ax=axes[0], label="Velocity norm (cm/s)")
        axes[0].set_title(f"{section.capitalize()} slice {slice_index} - Velocity field (t={time_frame})")
        axes[0].set_xlabel(xlabel)
        axes[0].set_ylabel(ylabel)
        plt.tight_layout()
        plt.show()

    def visualize_aliasing_simulation(self, axial_slice_index, coronal_slice_index, sagittal_slice_index, time_frame, protocol, aliased_pixels, velocity_post_aliasing, new_venc):
        """
        Visualize slices before and after aliasing simulation.
        """
        print(f"Starting to plot the aliased simulation...")
        sagittal_slice_non_aliased = self.get_slice_velocity(time_frame, "sagittal", protocol, sagittal_slice_index)
        coronal_slice_non_aliased = self.get_slice_velocity(time_frame, "coronal", protocol, coronal_slice_index)
        axial_slice_non_aliased = self.get_slice_velocity(time_frame, "axial", protocol, axial_slice_index)

        sagittal_slice_aliased = velocity_post_aliasing[:, :, sagittal_slice_index]
        coronal_slice_aliased = velocity_post_aliasing[coronal_slice_index, :, :]
        axial_slice_aliased = velocity_post_aliasing[:, axial_slice_index, :]

        fig, axes = plt.subplots(2, 3, figsize=(18, 12)) # 2 lines (one for non-aliased and one for aliased) and 3 columns (for the 3 different sections)

        # Line 1: non aliased
        axial_non_aliased_view = axes[0, 0].imshow(np.rot90(np.rot90(axial_slice_non_aliased)), cmap="gray", origin="lower")
        axes[0, 0].set_title(f"Axial Slice (Non-Aliased) at y={axial_slice_index}")

        coronal_non_aliased_view = axes[0, 1].imshow(np.rot90(np.rot90(np.rot90(coronal_slice_non_aliased.T))), cmap="gray", origin="lower")
        axes[0, 1].set_title(f"Coronal Slice (Non-Aliased) at x={coronal_slice_index}")

        sagittal_non_aliased_view = axes[0, 2].imshow(sagittal_slice_non_aliased.T, cmap="gray", origin="lower")
        axes[0, 2].set_title(f"Sagittal Slice (Non-Aliased) at z={sagittal_slice_index}")

        # Line 2: aliased
        axial_aliased_view = axes[1, 0].imshow(np.rot90(np.rot90(axial_slice_aliased)), cmap="gray", origin="lower")
        axes[1, 0].set_title(f"Axial Slice (Aliased with VENC={new_venc}) at y={axial_slice_index}")

        coronal_aliased_view = axes[1, 1].imshow(np.rot90(np.rot90(np.rot90(coronal_slice_aliased.T))), cmap="gray", origin="lower")
        axes[1, 1].set_title(f"Coronal Slice (Aliased with VENC={new_venc}) at x={coronal_slice_index}")

        sagittal_aliased_view = axes[1, 2].imshow(sagittal_slice_aliased.T, cmap="gray", origin="lower")
        axes[1, 2].set_title(f"Sagittal Slice (Aliased with VENC={new_venc}) at z={sagittal_slice_index}")

        fig.colorbar(axial_non_aliased_view, ax=axes[0,0], label="Velocity (cm/s)")
        fig.colorbar(coronal_non_aliased_view, ax=axes[0,1], label="Velocity (cm/s)")
        fig.colorbar(sagittal_non_aliased_view, ax=axes[0,2], label="Velocity (cm/s)")
        fig.colorbar(axial_aliased_view, ax=axes[1,0], label="Velocity (cm/s)")
        fig.colorbar(coronal_aliased_view, ax=axes[1,1], label="Velocity (cm/s)")
        fig.colorbar(sagittal_aliased_view, ax=axes[1,2], label="Velocity (cm/s)")

        # Add red dots we more or less intensity depending on the number of phase wraps of the pixel
        max_wraps = np.max(aliased_pixels)
        if max_wraps > 0:
            print(aliased_pixels.shape)
            aliased_pixels_axial_slice = np.rot90(np.rot90(aliased_pixels[:, axial_slice_index, :].T))
            print(aliased_pixels_axial_slice.shape)
            aliased_pixels_sagittal_slice = aliased_pixels[:, :, sagittal_slice_index]
            aliased_pixels_coronal_slice = np.rot90(np.rot90(np.rot90(aliased_pixels[coronal_slice_index, :, :])))
            for x in range(aliased_pixels_axial_slice.shape[0]):
                for z in range(aliased_pixels_axial_slice.shape[1]):
                    n_wraps = aliased_pixels_axial_slice[x,z]
                    if n_wraps > 0:
                        color_intensity = min(1.0, n_wraps / max_wraps) # color intensity function of the number of wraps
                        color = (color_intensity, 0, 0) # playing on the Red of RGB
                        axes[1, 0].scatter(x, z, color=color, s=2)
            for x in range(aliased_pixels_sagittal_slice.shape[0]):
                for y in range(aliased_pixels_sagittal_slice.shape[1]):
                    n_wraps = aliased_pixels_sagittal_slice[x,y]
                    if n_wraps > 0:
                        color_intensity = min(1.0, n_wraps / max_wraps) # color intensity function of the number of wraps
                        color = (color_intensity, 0, 0) # playing on the Red of RGB
                        axes[1, 2].scatter(x, y, color=color, s=2)
            for y in range(aliased_pixels_coronal_slice.shape[0]):
                for z in range(aliased_pixels_coronal_slice.shape[1]):
                    n_wraps = aliased_pixels_coronal_slice[y,z]
                    if n_wraps > 0:
                        color_intensity = min(1.0, n_wraps / max_wraps) # color intensity function of the number of wraps
                        color = (color_intensity, 0, 0) # playing on the Red of RGB
                        axes[1, 1].scatter(y, z, color=color, s=2)

        plt.tight_layout()
        plt.show()

    def simulate_aliasing(self, protocol, venc, time_frame):
        """
        Simultes the aliasing effect by virtually decreasing the VENC lower than the one of the acquisition to end up with velocities above +venc and under -venc to have aliased voxels
        protocol: the protocol on which the aliasing effect is simulated ("LR", "AP" or "SI")
        venc: the new velocity encoding (venc) value used, this value must be under the venc of the acquisition to end up with simulated aliased voxels
        time_frame: the time frame on which we want to apply the simulation
        """
        phase_data_before_aliasing = self.correspondance_nifti_data_protocol(protocol)[:,:,:,time_frame]
        phase_data_after_aliasing = np.zeros_like(phase_data_before_aliasing)
        velocity_before_aliasing = self.conversion_phase_to_velocity(protocol)[:,:,:,time_frame]
        velocity_post_aliasing = np.zeros_like(velocity_before_aliasing)
        acquisition_venc = self.get_venc(protocol)

        aliased_pixels = np.zeros_like(velocity_before_aliasing)

        print(f"Simulating the aliasing effect on the protocol {protocol} with VENC={venc} instead of original VENC (acquisition VENC)={acquisition_venc}")
        for x in range(velocity_before_aliasing.shape[0]):
            for y in range(velocity_before_aliasing.shape[1]):
                for z in range(velocity_before_aliasing.shape[2]):
                    pixel_velocity = velocity_before_aliasing[x,y,z]
                    while pixel_velocity > venc: 
                        velocity_post_aliasing[x,y,z] = pixel_velocity - 2*venc
                        pixel_velocity = velocity_post_aliasing[x,y,z]
                        aliased_pixels[x,y,z] += 1 # in each cell there will be the number of wraps of the aliased pixel
                    while pixel_velocity < -venc:
                        velocity_post_aliasing[x,y,z] = pixel_velocity + 2*venc
                        pixel_velocity = velocity_post_aliasing[x,y,z]
                        aliased_pixels[x,y,z] += 1
                    if pixel_velocity <= venc and pixel_velocity >= -venc:
                        velocity_post_aliasing[x,y,z] = pixel_velocity
                    phase_data_after_aliasing[x,y,z] = MAX_PIXEL_VALUE * velocity_post_aliasing[x,y,z] / venc # max_pixel_value * new_velocity_value / venc

        return velocity_post_aliasing, phase_data_after_aliasing, aliased_pixels

nifti_file = StudyCaseNIfTI("Dataset/IRM_BAO_069_1_4D_NIfTI")
#nifti_file.get_header("LR")
#nifti_file.get_image_shape_by_section("coronal","LR")
#nifti_file.get_number_of_slices_by_section("coronal","LR")
#nifti_file.get_number_of_time_frames("LR")
#nifti_file.visualize_slices(94, 189, 68,0,"LR")
#nifti_file.visualize_slice("coronal", 189, 0,"LR")
#nifti_file.get_venc("LR")
#nifti_file.visualize_velocity_vectors("sagittal", 73, 0, scale=1, stride=4)
#nifti_file.visualize_velocity_slice("sagittal", 73, 0, "LR")
velocity_post_aliasing, phase_data_after_aliasing, aliased_pixels = nifti_file.simulate_aliasing("LR", 50, 0)
nifti_file.visualize_aliasing_simulation(94, 189, 68, 0, "LR", aliased_pixels, velocity_post_aliasing, 50)