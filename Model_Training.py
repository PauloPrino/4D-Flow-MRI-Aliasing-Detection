import torch
import UNET
from torchvision import datasets, transforms
import torch.optim as optim
import MRIDataset
import matplotlib.pyplot as plt

num_epochs = 10

def dice_loss(model_output, masks, smooth=1.0): # model_output : the predicted mask by the model (an image of 0 and 1, non aliased and aliased pixels)
    intersection = (model_output * masks).sum(dim=(1, 2, 3)) # intersection between the predicted mask and the true mask
    dice = (2 * intersection + smooth) / (model_output.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3)) + smooth) # we sum on all channels, height, width because the tensor has the dimensions : batch size, channels, height, width
    return 1 - dice.mean()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNET.UNET(in_channels=1, out_channels=1, init_features=32).to(device)  # out_channels=1 pour segmentation binaire

criterion = dice_loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

data_preprocessing = transforms.Compose([
    transforms.Resize((256, 256)), # resizing to 256x256 to avoid using too much memory but still keep enough detail
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]), # normalizing
    # we can add here data augmentation on the training set
])

train_dataset = MRIDataset.MRIDataset(
    image_dir="",
    mask_dir="",
    transform=data_preprocessing
)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)

for epoch in range(num_epochs):
    model.train()
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        loss = criterion(outputs, masks) # loss which measures the ratio of overlapping correct predicted pixels

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
    val_loss = val_loss/len(val_loader) # average validation loss
    print(f"Validation loss: {val_loss:.4f}")

def plot_results(image, mask, pred_mask):
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title("MRI Image")
    plt.subplot(132)
    plt.imshow(mask.squeeze(), cmap='gray')
    plt.title("Grond truth mask")
    plt.subplot(133)
    plt.imshow(pred_mask.squeeze(), cmap='gray')
    plt.title("Predicted mask")
    plt.show()