import UNET_3D
import torch
import matplotlib.pyplot as plt
import numpy as np
import time

running_device = ""
if torch.cuda.is_available():
    print("CUDA available so running on GPU")
    running_device = "cuda"
else:
    print("CUDA not available so running on CPU")
    running_device = "cpu"

device = torch.device(running_device)

def transform_rotate_slice(slice, section_name):
        if section_name == "axial":
            return np.rot90(np.rot90(slice))
        if section_name == "coronal":
            return np.rot90(np.rot90(np.rot90(slice.T)))
        if section_name == "sagittal":
            return slice.T

def visualize_slices_and_masks(axial_slice_index, coronal_slice_index, sagittal_slice_index, predicted_mask_binary):
        """
        Visualize slices, ground truth mask and predicted mask
        """
        print(f"Starting to plot the aliased simulation...")
        volume_3D = np.load(volume_3D_path)
        ground_truth_mask = np.load(mask_path)

        sagittal_slice = volume_3D[:,:,sagittal_slice_index]
        coronal_slice = volume_3D[coronal_slice_index,:,:]
        axial_slice = volume_3D[:,axial_slice_index,:]

        sagittal_mask_predicted = predicted_mask_binary[:,:,sagittal_slice_index]
        coronal_mask_predicted = predicted_mask_binary[coronal_slice_index,:,:]
        axial_mask_predicted = predicted_mask_binary[:,axial_slice_index,:]

        sagittal_mask_ground_truth = ground_truth_mask[:,:,sagittal_slice_index]
        coronal_mask_ground_truth = ground_truth_mask[coronal_slice_index,:,:]
        axial_mask_ground_truth = ground_truth_mask[:,axial_slice_index,:]

        fig, axes = plt.subplots(3, 3, figsize=(18, 12)) # 3 lines (one for the slices and one for the predicted masks and one for the ground truth masks) and 3 columns (for the 3 different sections)

        # Line 1: slices
        axial_slice_view = axes[0, 0].imshow(transform_rotate_slice(axial_slice, "axial"), cmap="gray", origin="lower")
        axes[0, 0].set_title(f"Axial (Non-Aliased) at y={axial_slice_index}")

        coronal_slice_view = axes[0, 1].imshow(transform_rotate_slice(coronal_slice, "coronal"), cmap="gray", origin="lower")
        axes[0, 1].set_title(f"Coronal (Non-Aliased) at x={coronal_slice_index}")

        sagittal_slice_view = axes[0, 2].imshow(transform_rotate_slice(sagittal_slice, "sagittal"), cmap="gray", origin="lower")
        axes[0, 2].set_title(f"Sagittal (Non-Aliased) at z={sagittal_slice_index}")

        # Line 2: predicted masks
        axial_mask_predicted_view = axes[1, 0].imshow(transform_rotate_slice(axial_mask_predicted, "axial"), cmap="gray", origin="lower", vmin=0, vmax=1)
        axes[1, 0].set_title(f"Axial mask predicted at y={axial_slice_index}")

        coronal_mask_predicted_view = axes[1, 1].imshow(transform_rotate_slice(coronal_mask_predicted, "coronal"), cmap="gray", origin="lower", vmin=0, vmax=1)
        axes[1, 1].set_title(f"Coronal mask predicted at x={coronal_slice_index}")

        sagittal_mask_predicted_view = axes[1, 2].imshow(transform_rotate_slice(sagittal_mask_predicted, "sagittal"), cmap="gray", origin="lower", vmin=0, vmax=1)
        axes[1, 2].set_title(f"Sagittal mask predicted at z={sagittal_slice_index}")

        # Line 3: ground truth masks
        axial_mask_ground_truth_view = axes[2, 0].imshow(transform_rotate_slice(axial_mask_ground_truth, "axial"), cmap="gray", origin="lower", vmin=0, vmax=1)
        axes[2, 0].set_title(f"Axial mask ground truth at y={axial_slice_index}")

        coronal_mask_ground_truth_view = axes[2, 1].imshow(transform_rotate_slice(coronal_mask_ground_truth, "coronal"), cmap="gray", origin="lower", vmin=0, vmax=1)
        axes[2, 1].set_title(f"Coronal mask ground truth at x={coronal_slice_index}")

        sagittal_mask_ground_truth_view = axes[2, 2].imshow(transform_rotate_slice(sagittal_mask_ground_truth, "sagittal"), cmap="gray", origin="lower", vmin=0, vmax=1)
        axes[2, 2].set_title(f"Sagittal mask ground truth at z={sagittal_slice_index}")


        fig.colorbar(axial_slice_view, ax=axes[0,0], label="Phase value")
        fig.colorbar(coronal_slice_view, ax=axes[0,1], label="Phase value")
        fig.colorbar(sagittal_slice_view, ax=axes[0,2], label="Phase value")
        fig.colorbar(axial_mask_predicted_view, ax=axes[1,0], label="Phase value")
        fig.colorbar(coronal_mask_predicted_view, ax=axes[1,1], label="Phase value")
        fig.colorbar(sagittal_mask_predicted_view, ax=axes[1,2], label="Phase value")
        fig.colorbar(axial_mask_ground_truth_view, ax=axes[2,0], label="Phase value")
        fig.colorbar(coronal_mask_ground_truth_view, ax=axes[2,1], label="Phase value")
        fig.colorbar(sagittal_mask_ground_truth_view, ax=axes[2,2], label="Phase value")

        plt.tight_layout()
        plt.show()

def predict_aliased_pixels(model: UNET_3D.UNET_3D, input_3D_volume: str, ground_truth_mask:str, weight_file:str):
    """
    INPUT:
        - model: the model architecture we want to use 
        - input_3D_volume: the 3D volume on which we want to determine through our deep learning model which voxels are aliased
        - ground_truth_mask: the ground truth mask of aliased voxels
        - weight_file: the .pth file in which the weights are stored
    """
    start_time = time.time()
    model.load_state_dict(torch.load(weight_file, weights_only=True))
    model.eval()

    volume_3D = np.load(input_3D_volume).astype(np.float32) # loading the 3D volume that is saved as a numpy array and converting it to float32 as PyTorch uses floats and not int
    mask = np.load(ground_truth_mask).astype(np.float32)

    volume_3D = (volume_3D - volume_3D.min()) / (volume_3D.max() - volume_3D.min()) # normalize the values between 0 and 1: [0,1]

    volume_3D = torch.from_numpy(volume_3D) # convert to tensor
    mask = torch.from_numpy(mask)

    input_tensor = volume_3D.unsqueeze(0).unsqueeze(0).to(device) # twice unsqueeze because need to add dimension 1 for batch and for channels

    predicted_mask = torch.sigmoid(model(input_tensor))
    predicted_mask_binary = (predicted_mask > 0.5).float() # predicted binary mask so only 0 and 1 values
    predicted_mask_binary = predicted_mask_binary.squeeze().cpu().numpy() # we squeeze to not have the  dimension of channels and batches and put it to the cpu and go from a tensor to a numpy array
    print(f"Prediction time for one 3D volume at on time frame: {time.time() - start_time}")

    visualize_slices_and_masks(94, 189, 68, predicted_mask_binary)

model = UNET_3D.UNET_3D(in_channels=1, out_channels=1, init_features=8).to(device) # out_channels=1 for binary segmentation

volume_3D_path = "Dataset/CleanData/3DVolumes/IRM_BAO_069_1_4D_NIfTI_AP_t0.npy"
mask_path = "Dataset/CleanData/Masks/IRM_BAO_069_1_4D_NIfTI_AP_t0.npy"

predict_aliased_pixels(model, volume_3D_path, mask_path, "MRI_UNET_3D.pth")