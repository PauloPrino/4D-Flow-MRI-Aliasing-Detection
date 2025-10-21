import torch
import UNET_3D
from torchvision import datasets, transforms
import torch.optim as optim
import MRIDataset
import matplotlib.pyplot as plt
from torch.utils.data import random_split
import os
from PIL import Image
import numpy as np

num_epochs = 2

def dice_loss(model_output, masks, smooth=1.0): # model_output : the predicted mask by the model (an image of 0 and 1, non aliased and aliased pixels)
    probs = torch.sigmoid(model_output)
    intersection = (probs * masks).sum(dim=(1, 2, 3)) # intersection between the predicted mask and the true mask
    dice = (2 * intersection + smooth) / (probs.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3)) + smooth) # we sum on all channels, height, width because the tensor has the dimensions : batch size, channels, height, width
    return 1 - dice.mean()

bce = torch.nn.BCEWithLogitsLoss()

def combined_loss(logits, masks):
    dice = dice_loss(logits, masks)
    bce_loss = bce(logits, masks)
    return 0.5 * dice + 0.5 * bce_loss

def IoU_accuracy(pred_mask, true_mask):
    intersection = (pred_mask * true_mask).sum()
    union = pred_mask.sum() + true_mask.sum() - intersection

    if union == 0:
        return 0.0

    iou = intersection / union
    return iou.item() # scalar IoU value

running_device = ""
if torch.cuda.is_available():
    print("CUDA available so running on GPU")
    running_device = "cuda"
else:
    print("CUDA not available so running on CPU")
    running_device = "cpu"
device = torch.device(running_device)
model = UNET_3D.UNET_3D(in_channels=1, out_channels=1, init_features=8).to(device) # in_channels=1 because all in gray scale (no RGB), out_channels=1 for binary segmentation

criterion = combined_loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

full_dataset = MRIDataset.MRIDataset(
    volumes_3D_dir="Dataset/CleanData/3DVolumes", # folder containing all the 3D volumes
    mask_dir="Dataset/CleanData/Masks", # folder containing all the masks
)

total_size = len(full_dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset,
    [train_size, val_size, test_size],
)

print(f"Train size: {len(train_dataset)}")
print(f"Val size: {len(val_dataset)}")
print(f"Test size: {len(test_dataset)}")

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)

list_training_loss_by_epoch = []
list_validation_loss_by_epoch = []

list_training_accuracy_by_epoch = []
list_validation_accuracy_by_epoch = []

for epoch in range(num_epochs):
    IoU = 0
    print(f"Starting epoch {epoch + 1}/{num_epochs}")
    model.train()

    average_training_loss = 0
    i = 0
    for images, masks in train_loader: # do batch by batch the training
        i+=1
        
        images, masks = images.to(device), masks.to(device)
        #print(masks.min().item(), masks.max().item(), masks.unique()[:5])
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks) # loss which measures the ratio of overlapping correct predicted pixels
        average_training_loss += loss.item()
        print(f"Batch: {i}/{len(train_loader)}, training loss: {loss}")
        binary_output = (torch.sigmoid(outputs) > 0.5).float()
        IoU += IoU_accuracy(binary_output, masks)
        loss.backward()
        optimizer.step()
    average_training_loss = average_training_loss/len(train_loader)
    list_training_loss_by_epoch.append(average_training_loss)
    print(f"Average training loss over epoch {epoch + 1}: {average_training_loss}")
    IoU = IoU / len(train_loader)
    list_training_accuracy_by_epoch.append(IoU)
    print(f"Training IoU accuracy: {IoU*100}%")

    # Validation
    model.eval()
    val_loss = 0
    i = 0
    IoU = 0

    with torch.no_grad():
        for images, masks in val_loader:
            i += 1
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            print(f"Validation loss on batch {i}: {loss.item()}")
            val_loss += loss.item()
            binary_output = (torch.sigmoid(outputs) > 0.5).float()
            IoU += IoU_accuracy(binary_output, masks)
    val_loss = val_loss / i # average validation loss over all the validation batches
    list_validation_loss_by_epoch.append(val_loss)
    print(f"Average validation loss for all elements of the val loader: {val_loss:.4f}")
    IoU = IoU / i
    list_validation_accuracy_by_epoch.append(IoU)
    print(f"Validation IoU accuracy: {IoU*100}%")

# Testing once the model is trained
model.eval()
test_loss = 0
IoU = 0

with torch.no_grad():
    for images, masks in test_loader:
        images = images.to(device)
        masks = masks.to(device)
        outputs = model(images)
        test_loss += criterion(outputs, masks).item()
        binary_output = (torch.sigmoid(outputs) > 0.5).float()
        IoU += IoU_accuracy(binary_output, masks)
    test_loss = test_loss / len(test_loader)
    IoU = IoU / len(test_loader)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test IoU accuracy: {IoU*100}%")

torch.save(model.state_dict(), "MRI_UNET_3D.pth")

def plot_losses():
    x_epochs = []
    for i in range(num_epochs):
        x_epochs.append(i)
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.plot(x_epochs, list_training_loss_by_epoch, label='Training loss')
    plt.plot(x_epochs, list_validation_loss_by_epoch, label='Validation loss')
    plt.legend()

    plt.title('Evolution of losses through epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Losses')

plot_losses()

def plot_accuracies():
    x_epochs = []
    for i in range(num_epochs):
        x_epochs.append(i)
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.plot(x_epochs, list_training_accuracy_by_epoch, label='Training accuracy')
    plt.plot(x_epochs, list_validation_accuracy_by_epoch, label='Validation accuracy')
    plt.legend()

    plt.title('Evolution of accuracies through epochs')
    plt.xlabel('Epochs')
    plt.ylabel('IoU accuracy (%)')

plot_accuracies()

model.load_state_dict(torch.load("MRI_UNET_3D.pth", weights_only=True))
model.eval()

volume_3D_path = "Dataset/CleanData/3DVolumes/IRM_BAO_069_1_4D_NIfTI_AP_t0.npy"
mask_path = "Dataset/CleanData/Masks/IRM_BAO_069_1_4D_NIfTI_AP_t0.npy"

volume_3D = np.load(volume_3D_path).astype(np.float32) # loading the 3D volume that is saved as a numpy array and converting it to float32 as PyTorch uses floats and not int
mask = np.load(mask_path).astype(np.float32)

volume_3D = (volume_3D - volume_3D.min()) / (volume_3D.max() - volume_3D.min()) # normalize the values between 0 and 1: [0,1]

volume_3D = torch.from_numpy(volume_3D) # convert to tensor
mask = torch.from_numpy(mask)

input_tensor = volume_3D.unsqueeze(0).unsqueeze(0).to(device) # twice unsqueeze because need to add dimension 1 for batch and for channels

predicted_mask = torch.sigmoid(model(input_tensor))
predicted_mask_binary = (predicted_mask > 0.5).float() # predicted binary mask so only 0 and 1 values

def transform_rotate_slice(slice, section_name):
        if section_name == "axial":
            return np.rot90(np.rot90(slice))
        if section_name == "coronal":
            return np.rot90(np.rot90(np.rot90(slice.T)))
        if section_name == "sagittal":
            return slice.T

def visualize_slices_and_masks(axial_slice_index, coronal_slice_index, sagittal_slice_index):
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
        axial_mask_predicted_view = axes[1, 0].imshow(transform_rotate_slice(axial_mask_predicted, "axial"), cmap="gray", origin="lower")
        axes[1, 0].set_title(f"Axial mask predicted at y={axial_slice_index}")

        coronal_mask_predicted_view = axes[1, 1].imshow(transform_rotate_slice(coronal_mask_predicted, "coronal"), cmap="gray", origin="lower")
        axes[1, 1].set_title(f"Coronal mask predicted at x={coronal_slice_index}")

        sagittal_mask_predicted_view = axes[1, 2].imshow(transform_rotate_slice(sagittal_mask_predicted, "sagittal"), cmap="gray", origin="lower")
        axes[1, 2].set_title(f"Sagittal mask predicted at z={sagittal_slice_index}")

        # Line 3: ground truth masks
        axial_mask_ground_truth_view = axes[2, 0].imshow(transform_rotate_slice(axial_mask_ground_truth, "axial"), cmap="gray", origin="lower")
        axes[2, 0].set_title(f"Axial mask ground truth at y={axial_slice_index}")

        coronal_mask_ground_truth_view = axes[2, 1].imshow(transform_rotate_slice(coronal_mask_ground_truth, "coronal"), cmap="gray", origin="lower")
        axes[2, 1].set_title(f"Coronal mask ground truth at x={coronal_slice_index}")

        sagittal_mask_ground_truth_view = axes[2, 2].imshow(transform_rotate_slice(sagittal_mask_ground_truth, "sagittal"), cmap="gray", origin="lower")
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

visualize_slices_and_masks(94, 189, 68)