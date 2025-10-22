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

