import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import time
from matplotlib.animation import FuncAnimation, PillowWriter
import random
import json
import os

MAX_PIXEL_VALUE = 32768 # encoded in int16

class StudyCaseNIfTI():
    def __init__(self, study_path):
        """
        study_path: path to the folder of the study
        """
        self.study_path = study_path
        self.nifti_magnitude_path = study_path + "/" + os.path.basename(study_path) + ".nii.gz" # the file path of the magnitude (so the anatomy)
        self.nifti_LR_flow_path = study_path + "/" + os.path.basename(study_path) + "_e2.nii.gz" # the file path of the LR (Left-Right so axis x) of the blood flow
        self.nifti_AP_flow_path = study_path + "/" + os.path.basename(study_path) + "_e3.nii.gz" # the file path of the AP (Anterior-Posterior so axis y) of the blood flow
        self.nifti_SI_flow_path = study_path + "/" + os.path.basename(study_path) + "_e4.nii.gz" # the file path of the LR (Superior-Inferior so axis z) of the blood flow
        self.nifti_magnitude = nib.load(self.nifti_magnitude_path) # loading the NIfTI file
        self.nifti_LR_flow = nib.load(self.nifti_LR_flow_path)
        self.nifti_AP_flow = nib.load(self.nifti_AP_flow_path)
        self.nifti_SI_flow = nib.load(self.nifti_SI_flow_path)
        self.data_magnitude = np.asarray(self.nifti_magnitude.dataobj, dtype=np.uint16) # transform the image to a numpy array
        self.data_LR_flow = np.asarray(self.nifti_LR_flow.dataobj, dtype=np.int16) # dtype int16 to not take too much memory space (as int16 is only 2 bytes and the original files come in with phase values encoded in int16 for flows and uint16 for magnitude)
        self.data_AP_flow = np.asarray(self.nifti_AP_flow.dataobj, dtype=np.int16)
        self.data_SI_flow = np.asarray(self.nifti_SI_flow.dataobj, dtype=np.int16)
    
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
        
    def get_element_from_dicom_header(self, protocol, element_name):
        json_file = self.correspondance_nifti_file_path_protocol(protocol).replace(".nii.gz", "_dicom_header.json")

        with open(json_file, "r") as f:
            data = json.load(f)

        for key, value in data.items():
            if key == element_name:
                return value["value"]

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
    
    def get_3D_volume(self, time_frame, flow_direction):
        data = self.correspondance_nifti_data_protocol(flow_direction)
        return data[:,:,:,time_frame]

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
        venc = float(self.get_element_from_dicom_header(protocol, "[Velocity encoding]"))
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

    def visualize_anatomy(self, section, slice_index):
        slice = self.data_magnitude[slice_index, :, :, 0]

        fig, axes = plt.subplots(1, 1, figsize=(12, 4))

        view = axes.imshow(self.transform_rotate_slice(slice, section), cmap="gray", origin="lower")
        
        axes.set_title(f"{section} Slice at position ={slice_index}") # the pixels on the image are velocities in the direction perpendicular to the section

        fig.colorbar(view, ax=axes, label="Phase value")

        plt.tight_layout()
        plt.show()

    def visualize_slice(self, section, slice_index, time_frame, protocol):
        """
        Visualize a slice with the phase values
        """
        slice = self.get_slice(section, slice_index, time_frame, protocol)

        fig, axes = plt.subplots(1, 1, figsize=(12, 4))

        view = axes.imshow(self.transform_rotate_slice(slice, section), cmap="gray", origin="lower")
        
        axes.set_title(f"{section} Slice at position ={slice_index}") # the pixels on the image are velocities in the direction perpendicular to the section

        fig.colorbar(view, ax=axes, label="Phase value")

        plt.tight_layout()
        plt.show()

    def transform_rotate_slice(self, slice, section_name):
        if section_name == "axial":
            return np.rot90(np.rot90(slice))
        if section_name == "coronal":
            return np.rot90(np.rot90(np.rot90(slice.T)))
        if section_name == "sagittal":
            return slice.T

    def visualize_velocity_slice(self, section, slice_index, time_frame, protocol):
        """
        Visualize a slice with its velocity values and not phase values
        """
        slice_velocity = self.get_slice_velocity(time_frame, section, protocol, slice_index)      

        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(self.transform_rotate_slice(slice_velocity, section), cmap="seismic", origin="lower")
        ax.set_title(f"{section.capitalize()} slice {slice_index} - Velocity (cm/s)")
        cbar = fig.colorbar(im, ax=ax, label="Velocity (cm/s)")
        plt.show()

    def create_mask_from_magnitude(self, magnitude_data, threshold=0.1):
        mask = magnitude_data > threshold
        return mask.astype(np.uint8)


    def conversion_phase_to_velocity(self, protocol):
        """
        Converts phase data (direct data comming from the MRI machine) to velocity data using the venc value
        """
        venc = float(self.get_element_from_dicom_header(protocol, "[Velocity encoding]"))
        protocol_data = self.correspondance_nifti_data_protocol(protocol).astype(np.float32)
        print(f"Converting phase pixel values into velocities for protocol {protocol} using VENC={venc}cm/s")
        velocities = (protocol_data / MAX_PIXEL_VALUE) * venc
        return velocities

    def visualize_velocity_vectors(self, section, slice_index, interval, scale=2, stride=4, time_frame=0, mask=True):
        """
        Visualize a slice with velocity vectors (arrows) overlaid on top of the magnitude (anatomy)
        section: 'axial', 'coronal', or 'sagittal'
        slice_index: index of the slice in that section
        interval: time in ms between the rendering of two consecutive images
        scale: scaling factor for arrow length
        stride: skip pixels to avoid overcrowding, not draw an arrow on each pixel
        time_frame: the time frame to be visualised or if time_frame=0 then the animation is visualised
        """
        mag = self.data_magnitude
        
        vx = self.conversion_phase_to_velocity("LR") # velocity in nifti
        vy = self.conversion_phase_to_velocity("AP")
        vz = self.conversion_phase_to_velocity("SI")
        
        if mask:
            magnitude_mask = self.create_mask_from_magnitude(mag, 300)
            vx = vx * magnitude_mask
            vy = vy * magnitude_mask
            vz = vz * magnitude_mask

        if section == "axial":
            magnitude_data = mag[:, slice_index, :, :]
            u_data = vx[:, slice_index, :, :]
            v_data = vy[:, slice_index, :, :]
            xlabel, ylabel = "Left-Right", "Anterior-Posterior"

        elif section == "coronal":
            magnitude_data = mag[slice_index, :, :, :]
            u_data = vx[slice_index, :, :, :]
            v_data = vz[slice_index, :, :, :]
            xlabel, ylabel = "Left-Right", "Superior-Inferior"

        elif section == "sagittal":
            magnitude_data = mag[:, :, slice_index, :]
            u_data = vy[:, :, slice_index, :]
            v_data = vz[:, :, slice_index, :]
            xlabel, ylabel = "Anterior-Posterior", "Superior-Inferior"

        # Downsample to avoid too many arrows (we do some striding similar as we do on )
        u = self.transform_rotate_slice(u_data[::stride, ::stride, time_frame], section)
        v = self.transform_rotate_slice(v_data[::stride, ::stride, time_frame], section)
        mag = self.transform_rotate_slice(magnitude_data[:, :, time_frame], section)

        vector_norm = np.sqrt(u**2 + v**2) # norm of the velocity

        # Generate coordinates for the arrows, also downsampled
        Y, X = np.mgrid[0:mag.shape[0]:stride, 0:mag.shape[1]:stride]

        fig, ax = plt.subplots(1, 1, figsize=(12,4))
        anatomical_view = ax.imshow(mag, cmap='gray', origin='lower')

        norm = plt.Normalize(vmin=np.min(vector_norm), vmax=np.max(vector_norm))
        cmap = plt.colormaps['plasma']
        quiv = ax.quiver( # overlay arrows on top of the image
            X, Y, u, -v,
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
        fig.colorbar(quiv, ax=ax, label="Velocity norm (cm/s)")
        ax.set_title(f"{section.capitalize()} slice {slice_index} - Velocity field (t={time_frame})")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        def update(frame):
            u = self.transform_rotate_slice(u_data[::stride, ::stride, frame], section)
            v = self.transform_rotate_slice(v_data[::stride, ::stride, frame], section)
            mag_frame = self.transform_rotate_slice(magnitude_data[:, :, frame], section)

            vector_norm = np.sqrt(u**2 + v**2)
            quiv.set_UVC(u, -v, vector_norm) # updates the quiver plot (the vector fields) to be the one of the current time frame
            anatomical_view.set_data(mag_frame) # update the data of the magnitude to be the one of the current time frame
            ax.set_title(f"{section.capitalize()} slice {slice_index} - Velocity field (t={frame})")
            return quiv, anatomical_view

        if time_frame == 0: # if we want an animation
            anim = FuncAnimation(fig, update, frames=50, interval=interval, blit=False)
            anim.save(f"velocity_vectors_animation_{section}_slice_{slice_index}_mask_{mask}.gif", writer=PillowWriter(fps=interval))
        plt.tight_layout()
        plt.show()


    def visualize_aliasing_simulation(self, axial_slice_index, coronal_slice_index, sagittal_slice_index, time_frame, interval, flow_direction, aliased_pixels, velocity_post_aliasing, new_venc):
        """
        Visualize slices before and after aliasing simulation.
        """
        print(f"Starting to plot the aliased simulation...")
        sagittal_slice_non_aliased = self.get_slice_velocity(time_frame, "sagittal", flow_direction, sagittal_slice_index)
        coronal_slice_non_aliased = self.get_slice_velocity(time_frame, "coronal", flow_direction, coronal_slice_index)
        axial_slice_non_aliased = self.get_slice_velocity(time_frame, "axial", flow_direction, axial_slice_index)

        sagittal_slice_aliased = velocity_post_aliasing[:, :, sagittal_slice_index, time_frame]
        coronal_slice_aliased = velocity_post_aliasing[coronal_slice_index, :, :, time_frame]
        axial_slice_aliased = velocity_post_aliasing[:, axial_slice_index, :, time_frame]

        fig, axes = plt.subplots(3, 3, figsize=(18, 12)) # 2 lines (one for non-aliased and one for aliased) and 3 columns (for the 3 different sections)

        # Line 1: non aliased
        axial_non_aliased_view = axes[0, 0].imshow(self.transform_rotate_slice(axial_slice_non_aliased, "axial"), cmap="gray", origin="lower")
        axes[0, 0].set_title(f"Axial (Non-Aliased) at y={axial_slice_index} at t={time_frame}")

        coronal_non_aliased_view = axes[0, 1].imshow(self.transform_rotate_slice(coronal_slice_non_aliased, "coronal"), cmap="gray", origin="lower")
        axes[0, 1].set_title(f"Coronal (Non-Aliased) at x={coronal_slice_index} at t={time_frame}")

        sagittal_non_aliased_view = axes[0, 2].imshow(self.transform_rotate_slice(sagittal_slice_non_aliased, "sagittal"), cmap="gray", origin="lower")
        axes[0, 2].set_title(f"Sagittal (Non-Aliased) at z={sagittal_slice_index} at t={time_frame}")

        # Line 2: aliased
        axial_aliased_view = axes[1, 0].imshow(self.transform_rotate_slice(axial_slice_aliased, "axial"), cmap="gray", origin="lower")
        axes[1, 0].set_title(f"Axial (Aliased with VENC={new_venc}) at y={axial_slice_index} at t={time_frame}")

        coronal_aliased_view = axes[1, 1].imshow(self.transform_rotate_slice(coronal_slice_aliased, "coronal"), cmap="gray", origin="lower")
        axes[1, 1].set_title(f"Coronal (Aliased with VENC={new_venc}) at x={coronal_slice_index} at t={time_frame}")

        sagittal_aliased_view = axes[1, 2].imshow(self.transform_rotate_slice(sagittal_slice_aliased, "sagittal"), cmap="gray", origin="lower")
        axes[1, 2].set_title(f"Sagittal (Aliased with VENC={new_venc}) at z={sagittal_slice_index} at t={time_frame}")

        # Line 3: aliased on which we'll put the red dots to indicate aliased pixels
        axial_aliased_with_dots_view = axes[2, 0].imshow(self.transform_rotate_slice(axial_slice_aliased, "axial"), cmap="gray", origin="lower")
        axes[2, 0].set_title(f"Axial (Aliased with VENC={new_venc}) at y={axial_slice_index} at t={time_frame}")

        coronal_aliased_with_dots_view = axes[2, 1].imshow(self.transform_rotate_slice(coronal_slice_aliased, "coronal"), cmap="gray", origin="lower")
        axes[2, 1].set_title(f"Coronal (Aliased with VENC={new_venc}) at x={coronal_slice_index} at t={time_frame}")

        sagittal_aliased_with_dots_view = axes[2, 2].imshow(self.transform_rotate_slice(sagittal_slice_aliased, "sagittal"), cmap="gray", origin="lower")
        axes[2, 2].set_title(f"Sagittal (Aliased with VENC={new_venc}) at z={sagittal_slice_index} at t={time_frame}")

        fig.colorbar(axial_non_aliased_view, ax=axes[0,0], label="Velocity (cm/s)")
        fig.colorbar(coronal_non_aliased_view, ax=axes[0,1], label="Velocity (cm/s)")
        fig.colorbar(sagittal_non_aliased_view, ax=axes[0,2], label="Velocity (cm/s)")
        fig.colorbar(axial_aliased_view, ax=axes[1,0], label="Velocity (cm/s)")
        fig.colorbar(coronal_aliased_view, ax=axes[1,1], label="Velocity (cm/s)")
        fig.colorbar(sagittal_aliased_view, ax=axes[1,2], label="Velocity (cm/s)")
        fig.colorbar(axial_aliased_view, ax=axes[2,0], label="Velocity (cm/s)")
        fig.colorbar(coronal_aliased_view, ax=axes[2,1], label="Velocity (cm/s)")
        fig.colorbar(sagittal_aliased_view, ax=axes[2,2], label="Velocity (cm/s)")

        # Add red dots we more or less intensity depending on the number of phase wraps of the pixel
        def plot_red_dots(aliased_pixels, axes, axial_slice_index, coronal_slice_index, sagittal_slice_index):
            max_wraps = np.max(aliased_pixels)
            if max_wraps > 0:
                # Axial
                aliased_pixels_axial_slice = self.transform_rotate_slice(aliased_pixels[:, axial_slice_index, :], "axial")
                y_axial, x_axial = np.where(aliased_pixels_axial_slice > 0)
                n_wraps_axial = aliased_pixels_axial_slice[y_axial, x_axial]
                color_intensity_axial = np.minimum(1.0, n_wraps_axial / max_wraps)
                colors_axial = np.column_stack([color_intensity_axial, np.zeros_like(color_intensity_axial), np.zeros_like(color_intensity_axial)])
                axes[2, 0].scatter(x_axial, y_axial, color=colors_axial, s=2)

                # Coronal
                aliased_pixels_coronal_slice = self.transform_rotate_slice(aliased_pixels[coronal_slice_index, :, :], "coronal")
                y_coronal, x_coronal = np.where(aliased_pixels_coronal_slice > 0)
                n_wraps_coronal = aliased_pixels_coronal_slice[y_coronal, x_coronal]
                color_intensity_coronal = np.minimum(1.0, n_wraps_coronal / max_wraps)
                colors_coronal = np.column_stack([color_intensity_coronal, np.zeros_like(color_intensity_coronal), np.zeros_like(color_intensity_coronal)])
                axes[2, 1].scatter(x_coronal, y_coronal, color=colors_coronal, s=2)

                # Sagittal
                aliased_pixels_sagittal_slice = self.transform_rotate_slice(aliased_pixels[:, :, sagittal_slice_index], "sagittal")
                y_sagittal, x_sagittal = np.where(aliased_pixels_sagittal_slice > 0)
                n_wraps_sagittal = aliased_pixels_sagittal_slice[y_sagittal, x_sagittal]
                color_intensity_sagittal = np.minimum(1.0, n_wraps_sagittal / max_wraps)
                colors_sagittal = np.column_stack([color_intensity_sagittal, np.zeros_like(color_intensity_sagittal), np.zeros_like(color_intensity_sagittal)])
                axes[2, 2].scatter(x_sagittal, y_sagittal, color=colors_sagittal, s=2)

        plot_red_dots(aliased_pixels[:,:,:,time_frame], axes, axial_slice_index, coronal_slice_index, sagittal_slice_index)


        def update(frame):
            print(f"Rendering frame {frame}")
            sagittal_slice_non_aliased = self.get_slice_velocity(frame, "sagittal", flow_direction, sagittal_slice_index)
            coronal_slice_non_aliased = self.get_slice_velocity(frame, "coronal", flow_direction, coronal_slice_index)
            axial_slice_non_aliased = self.get_slice_velocity(frame, "axial", flow_direction, axial_slice_index)

            sagittal_slice_aliased = velocity_post_aliasing[:, :, sagittal_slice_index, frame]
            coronal_slice_aliased = velocity_post_aliasing[coronal_slice_index, :, :, frame]
            axial_slice_aliased = velocity_post_aliasing[:, axial_slice_index, :, frame]
            # Update non-aliased images
            axial_non_aliased_view.set_data(self.transform_rotate_slice(axial_slice_non_aliased, "axial"))
            axes[0,0].set_title(f"Axial (Non-Aliased) at y={axial_slice_index} at t={frame}")
            coronal_non_aliased_view.set_data(self.transform_rotate_slice(coronal_slice_non_aliased, "coronal"))
            axes[0,1].set_title(f"Coronal (Non-Aliased) at x={coronal_slice_index} at t={frame}")
            sagittal_non_aliased_view.set_data(self.transform_rotate_slice(sagittal_slice_non_aliased, "sagittal"))
            axes[0,2].set_title(f"Sagittal (Non-Aliased) at z={sagittal_slice_index} at t={frame}")

            # Update aliased images
            axial_aliased_view.set_data(self.transform_rotate_slice(axial_slice_aliased, "axial"))
            axes[1,0].set_title(f"Axial (Aliased with VENC={new_venc}) at y={axial_slice_index} at t={frame}")
            coronal_aliased_view.set_data(self.transform_rotate_slice(coronal_slice_aliased, "coronal"))
            axes[1,1].set_title(f"Coronal (Aliased with VENC={new_venc}) at x={coronal_slice_index} at t={frame}")
            sagittal_aliased_view.set_data(self.transform_rotate_slice(sagittal_slice_aliased, "sagittal"))
            axes[1,2].set_title(f"Sagittal (Aliased with VENC={new_venc}) at z={sagittal_slice_index} at t={frame}")

            # Update aliased images with red dots
            axial_aliased_with_dots_view.set_data(self.transform_rotate_slice(axial_slice_aliased, "axial"))
            axes[2,0].set_title(f"Axial (Aliased with VENC={new_venc}) at y={axial_slice_index} at t={frame}")
            coronal_aliased_with_dots_view.set_data(self.transform_rotate_slice(coronal_slice_aliased, "coronal"))
            axes[2,1].set_title(f"Coronal (Aliased with VENC={new_venc}) at x={coronal_slice_index} at t={frame}")
            sagittal_aliased_with_dots_view.set_data(self.transform_rotate_slice(sagittal_slice_aliased, "sagittal"))
            axes[2,2].set_title(f"Sagittal (Aliased with VENC={new_venc}) at z={sagittal_slice_index} at t={frame}")

            plot_red_dots(aliased_pixels[:,:,:,frame], axes, axial_slice_index, coronal_slice_index, sagittal_slice_index)

            return axial_non_aliased_view, coronal_non_aliased_view, sagittal_non_aliased_view, \
                axial_aliased_view, coronal_aliased_view, sagittal_aliased_view, \
                axial_aliased_with_dots_view, coronal_aliased_with_dots_view, sagittal_aliased_with_dots_view

        if time_frame == 0: # if we want an animation
            anim = FuncAnimation(fig, update, frames=50, interval=interval, blit=False)
            anim.save(f"aliasing_simulation_animation.gif", writer=PillowWriter(fps=interval))

        #plt.tight_layout()
        #plt.show()

    def simulate_aliasing(self, flow_direction, venc, time_frame):
        """
        Simultes the aliasing effect by virtually decreasing the VENC lower than the one of the acquisition to end up with velocities above +venc and under -venc to have aliased voxels
        flow_direction: the flow_direction on which the aliasing effect is simulated ("LR", "AP" or "SI")
        venc: the new velocity encoding (venc) value used, this value must be under the venc of the acquisition to end up with simulated aliased voxels
        time_frame: the time frame on which we want to apply the simulation
        """
        starting_time = time.time()
        if time_frame:
            phase_data_before_aliasing = self.correspondance_nifti_data_protocol(flow_direction)[:,:,:,time_frame]
            velocity_before_aliasing = self.conversion_phase_to_velocity(flow_direction)[:,:,:,time_frame]
        else:
            phase_data_before_aliasing = self.correspondance_nifti_data_protocol(flow_direction)
            velocity_before_aliasing = self.conversion_phase_to_velocity(flow_direction)

        phase_data_after_aliasing = np.zeros_like(phase_data_before_aliasing)
        velocity_post_aliasing = np.zeros_like(velocity_before_aliasing) # float32
        acquisition_venc = float(self.get_element_from_dicom_header(flow_direction, "[Velocity encoding]"))

        aliased_pixels = np.floor((np.abs(velocity_before_aliasing) + venc) / (2 * venc)).astype(bool) # numpy array of the number of wraps for each pixel (so value=0 if no wraps so if not aliased), we precise the type bool to take only 1 byte

        print(f"Simulating the aliasing effect on the flow direction {flow_direction} with VENC={venc} instead of original VENC (acquisition VENC)={acquisition_venc}")
        
        velocity_post_aliasing = np.where(
                velocity_before_aliasing > venc,
                velocity_before_aliasing - 2 * venc * aliased_pixels, # we wrap the number of times as necessary
                np.where(
                    velocity_before_aliasing < -venc,
                    velocity_before_aliasing + 2 * venc * aliased_pixels,
                    velocity_before_aliasing
                )
            )
        
        phase_data_after_aliasing = (velocity_post_aliasing * MAX_PIXEL_VALUE / venc).astype(np.int16)
                
        duration = time.time() - starting_time
        print(f"Duration of the simulation: {duration}s")
        return velocity_post_aliasing, aliased_pixels, phase_data_after_aliasing # aliased pixels is the mask of the aliased pixels with values equal to the number of wraps (value=1 if pixel aliased otherwise value=0)

#nifti_file = StudyCaseNIfTI("Dataset/RawData/IRM_BAO_069_1_4D_NIfTI")
#nifti_file.get_header("LR")
#nifti_file.get_image_shape_by_section("coronal","LR")
#nifti_file.get_number_of_slices_by_section("coronal","LR")
#nifti_file.get_number_of_time_frames("LR")
#nifti_file.visualize_slices(94, 189, 68,0,"LR")
#nifti_file.visualize_slice("coronal", 189, 0,"LR")
#print(f"Velocity encoding: {nifti_file.get_element_from_dicom_header("LR", "[Velocity encoding]")}")
#nifti_file.visualize_velocity_vectors("coronal", 160, 100, scale=2, stride=4, time_frame=0, mask=True)
#nifti_file.visualize_velocity_slice("coronal", 175, 0, "LR")
#velocity_post_aliasing, aliased_pixels, phase_data_after_aliasing = nifti_file.simulate_aliasing("LR", 50, None)
#nifti_file.visualize_aliasing_simulation(94, 189, 68, 0, 100, "LR", aliased_pixels, velocity_post_aliasing, 50)
#nifti_file.visualize_anatomy("coronal", 166)