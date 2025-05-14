import os
import torch
import torch.nn as nn
import pydicom
import numpy as np
from django.shortcuts import render
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.http import JsonResponse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from django.conf import settings
from PIL import Image
import random
import torchvision.transforms as transforms


# Check and create necessary directories
def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

# ==================== SEBlock ====================
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch, channels, _, _ = x.size()
        y = self.avg_pool(x).view(batch, channels)
        y = self.fc(y).view(batch, channels, 1, 1)
        return x * y

# ==================== EnhancedRedCNN ====================
class EnhancedRedCNN(nn.Module):
    def __init__(self):
        super(EnhancedRedCNN, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(1, 64, 5, 1, padding=2)    # 1 -> 64
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 5, 1, padding=2)  # 64 -> 128
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 5, 1, padding=2) # 128 -> 256
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, 5, 1, padding=2) # 256 -> 256
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 256, 5, 1, padding=2) # 256 -> 256
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 128, 5, 1, padding=2) # 256 -> 128
        self.bn6 = nn.BatchNorm2d(128)
        self.conv7 = nn.Conv2d(128, 64, 5, 1, padding=2)  # 128 -> 64
        self.bn7 = nn.BatchNorm2d(64)

        # Attention Blocks
        self.attention1 = SEBlock(256)  # After conv3 (256 channels)
        self.attention2 = SEBlock(256)  # After conv5 (256 channels)
        self.attention3 = SEBlock(64)   # After conv7 (64 channels)

        # Decoder (symmetric to encoder)
        self.deconv1 = nn.ConvTranspose2d(64, 128, 5, 1, padding=2)   # 64 -> 128
        self.bn_d1 = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(128, 256, 5, 1, padding=2)  # 128 -> 256
        self.bn_d2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, 5, 1, padding=2)  # 256 -> 128 (matches residual2)
        self.bn_d3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 256, 5, 1, padding=2)  # 128 -> 256
        self.bn_d4 = nn.BatchNorm2d(256)
        self.deconv5 = nn.ConvTranspose2d(256, 256, 5, 1, padding=2)  # 256 -> 256 (matches residual3)
        self.bn_d5 = nn.BatchNorm2d(256)
        self.deconv6 = nn.ConvTranspose2d(256, 64, 5, 1, padding=2)   # 256 -> 64
        self.bn_d6 = nn.BatchNorm2d(64)
        self.deconv_last = nn.ConvTranspose2d(64, 1, 5, 1, padding=2) # 64 -> 1 (matches residual1)

        self.relu = nn.ReLU()

    def forward(self, x):
        # Encoder
        residual1 = x  # 1 channel
        x = self.relu(self.bn1(self.conv1(x)))  # 64 channels
        x = self.relu(self.bn2(self.conv2(x)))  # 128 channels
        residual2 = x  # 128 channels
        x = self.relu(self.bn3(self.conv3(x)))  # 256 channels
        x = self.attention1(x)
        x = self.relu(self.bn4(self.conv4(x)))  # 256 channels
        x = self.relu(self.bn5(self.conv5(x)))  # 256 channels
        x = self.attention2(x)
        residual3 = x  # 256 channels
        x = self.relu(self.bn6(self.conv6(x)))  # 128 channels
        x = self.relu(self.bn7(self.conv7(x)))  # 64 channels
        x = self.attention3(x)  # 64 channels

        # Decoder
        x = self.relu(self.bn_d1(self.deconv1(x)))  # 128 channels
        x = self.relu(self.bn_d2(self.deconv2(x)))  # 256 channels
        x = self.relu(self.bn_d3(self.deconv3(x)) + residual2)  # 128 channels + 128 channels
        x = self.relu(self.bn_d4(self.deconv4(x)))  # 256 channels
        x = self.relu(self.bn_d5(self.deconv5(x)) + residual3)  # 256 channels + 256 channels
        x = self.relu(self.bn_d6(self.deconv6(x)))  # 64 channels
        x = self.deconv_last(x) + residual1  # 1 channel + 1 channel

        return x

# Load the single EnhancedRedCNN model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_base_path = r"C:\Users\tharu\OneDrive\Desktop\VanderLindes\VanderLindes\redcnn"
model_path = os.path.join(model_base_path, "best_enhanced_red_cnn_16.pth")
red_cnn_model = EnhancedRedCNN().to(device)
try:
    red_cnn_model.load_state_dict(torch.load(model_path, map_location=device))
    red_cnn_model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {str(e)}")

def load_dicom_image(dicom_path):
    dicom_data = pydicom.dcmread(dicom_path)
    image = dicom_data.pixel_array.astype(np.float32)
    return dicom_data, image

def save_dicom_image(original_dicom, modified_image, save_path):
    ensure_directory(os.path.dirname(save_path))
    new_dicom = original_dicom.copy()
    modified_image = np.clip(modified_image, 0, np.iinfo(original_dicom.pixel_array.dtype).max).astype(original_dicom.pixel_array.dtype)
    new_dicom.PixelData = modified_image.tobytes()
    new_dicom.save_as(save_path)

def save_as_jpg(image_array, save_path):
    ensure_directory(os.path.dirname(save_path))
    min_val, max_val = np.min(image_array), np.max(image_array)
    image_array = (image_array - min_val) / (max_val - min_val) * 255.0
    Image.fromarray(np.uint8(image_array)).save(save_path, format="JPEG")

def model(request):
    data = {}

    if request.method == "POST" and request.FILES.get("dicom_file"):
        try:
            # 1) Save uploaded noisy scan
            uploaded_file = request.FILES["dicom_file"]
            name = uploaded_file.name
            ext = os.path.splitext(name)[1].lower()
            if ext not in [".dcm", ".ima"]:
                raise ValueError("Invalid file format. Please upload a DCM or IMA file.")

            noisy_name = default_storage.save(
                os.path.join(settings.NOISY_DIR, name),
                ContentFile(uploaded_file.read())
            )
            noisy_path = default_storage.path(noisy_name)

            # 2) Inline denoising
            with torch.no_grad():
                # load + normalize noisy image
                noisy_img = pydicom.dcmread(noisy_path).pixel_array.astype(np.float32)
                noisy_img = (noisy_img - noisy_img.min()) / (noisy_img.max() - noisy_img.min() + 1e-8)

                # to tensor, resize, to device
                tensor = torch.tensor(noisy_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                tensor = transforms.Resize((512, 512))(tensor).to(device)

                # forward pass
                denoised_tensor = red_cnn_model(tensor)
                denoised = denoised_tensor.squeeze().cpu().numpy()

            # 3) Save outputs (JPEG + DICOM)
            denoised_255 = denoised * 255.0
            save_dicom_image(
                pydicom.dcmread(noisy_path),
                denoised_255,
                os.path.join(settings.DENOISED_DIR, name)
            )

            noisy_jpg = os.path.join(settings.NOISY_JPG_DIR, name.replace(ext, ".jpg"))
            denoised_jpg = os.path.join(settings.DENOISED_JPG_DIR, name.replace(ext, ".jpg"))
            save_as_jpg(noisy_img, noisy_jpg)
            save_as_jpg(denoised, denoised_jpg)

            # 4) Compute metrics between noisy and denoised (both in [0..1])
            psnr_val = psnr(noisy_img, denoised, data_range=1.0)
            ssim_val = ssim(noisy_img, denoised, data_range=1.0)
            sig_pwr = np.mean(noisy_img**2)
            noise_pwr = np.mean((noisy_img - denoised)**2)
            snr_val = 10 * np.log10(sig_pwr / noise_pwr) if noise_pwr > 0 else float("inf")

            # 5) Pack data for the template
            data = {
                "message": "Processing complete",
                "denoised_image": os.path.join(settings.MEDIA_URL, "denoised_jpg/", os.path.basename(denoised_jpg)),
                "noisy_image":    os.path.join(settings.MEDIA_URL, "noisy_jpg/",    os.path.basename(noisy_jpg)),
                "denoised_dicom_path": os.path.join(settings.MEDIA_URL, "denoised/", name),
                "denoised_PSNR": f"{psnr_val:.2f} dB",
                "denoised_SSIM": f"{ssim_val:.4f}",
                "denoised_SNR":  f"{snr_val:.2f} dB"
            }

        except Exception as exc:
            data = {"error": f"❌ Processing failed: {exc}"}

    return render(request, "model.html", {"data": data})


