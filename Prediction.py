import UNET
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms

data_preprocessing = transforms.Compose([
    transforms.Resize((256, 256)), # resizing to 256x256 to avoid using too much memory but still keep enough detail
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), # normalizing
])

running_device = ""
if torch.cuda.is_available():
    print("CUDA available so running on GPU")
    running_device = "cuda"
else:
    print("CUDA not available so running on CPU")
    running_device = "cpu"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNET.UNET(in_channels=3, out_channels=1, init_features=32).to(device) # out_channels=1 for binary segmentation

model.load_state_dict(torch.load("unet_model.pth", weights_only=True))
model.eval()

def plot_results(image, mask, pred_mask):
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title("MRI Image")
    plt.subplot(132)
    plt.imshow(mask.squeeze(), cmap='gray')
    plt.title("Grond truth mask")
    plt.subplot(133)
    plt.imshow(pred_mask.detach().squeeze().cpu().numpy(), cmap='gray')
    plt.title("Predicted mask")
    plt.show()

image_np = plt.imread("swimseg-2/test/0913.png") # image loaded as a numpy array
mask_np = plt.imread("swimseg-2/test_labels/0913.png")

# If the image is float, convert to uint8
if image_np.dtype == np.float32 or image_np.dtype == np.float64:
    image_np = (255 * image_np).astype(np.uint8)
if mask_np.dtype == np.float32 or mask_np.dtype == np.float64:
    mask_np = (255 * mask_np).astype(np.uint8)

image = Image.fromarray(image_np) # converting from numpy to PIL

input_tensor = data_preprocessing(image).unsqueeze(0).to(device)

predicted_mask = torch.sigmoid(model(input_tensor))
predicted_mask_binary = (predicted_mask > 0.5).float() # predicted binary mask so only 0 and 1 values
plot_results(image_np, mask_np, predicted_mask_binary.cpu())