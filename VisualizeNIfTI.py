import nibabel as nib
import matplotlib.pyplot as plt

def visualize_nifti_slices(nii_gz_path, time_frame, position_x, position_y, position_z):
    """
    nii_gz_path: path to the .nii.gz file
    time_frame: time frame to be visualized (value between 0 and 49, because there are 49 time frames)
    position_x: position of the sagittal slice
    """
    img = nib.load(nii_gz_path) # loading the NIfTI file
    data = img.get_fdata() # transform the image to a numpy array

    print(data.shape)
    image_shape = data.shape[0]
    print(f"Image size: {image_shape} x {image_shape}")
    number_of_slices = data.shape[2]
    print(f"Number of slices in each direction: {number_of_slices}")
    number_of_frames = data.shape[3]
    print(f"Number of time frames: {number_of_frames}")
    data = data[:, :, :, time_frame] # choose time frame to be viualized
    print(data[position_x,:,:])
    print(f"Image shape: {data.shape}")

    # We only take one slice in each direction at the position asked
    slice_x = data[position_x, :, :] # sagital slice
    slice_y = data[:, position_y, :] # coronal slice
    slice_z = data[:, :, position_z] # axial slice

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(slice_x.T, cmap="gray", origin="lower")
    axes[0].set_title(f"Axial Slice at poistion z={position_z}")

    axes[1].imshow(slice_y.T, cmap="gray", origin="lower")
    axes[1].set_title(f"Coronal Slice at position y={position_y}")

    axes[2].imshow(slice_z.T, cmap="gray", origin="lower")
    axes[2].set_title(f"Sagital Slice at position x={position_x}")

    plt.tight_layout()
    plt.show()

nii_path = "Dataset/IRM_BAO_069_1_4D_NIfTI/IRM_BAO_069_1_4D_NIfTI_e3.nii.gz"
visualize_nifti_slices(nii_path, time_frame=5, position_x=64, position_y=64, position_z=32)
#visualize_nifti_slices(nii_path, time_frame=40, position_x=64, position_y=64, position_z=32)