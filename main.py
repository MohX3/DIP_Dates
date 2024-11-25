import os
import cv2
import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import torch
import torch.nn as nn
from bm3d import bm3d
from skimage.restoration import estimate_sigma, wiener
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Define the DnCNN model
class DnCNN(nn.Module):
    def __init__(self, channels=3, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(channels, features, kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers - 2):
            layers.append(nn.Conv2d(features, features, kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(features, channels, kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
    def forward(self, x):
        out = self.dncnn(x)
        return x - out  # Residual learning

# Function to denoise image using BM3D
def denoise_image_with_bm3d(img):
    # BM3D works with images in range [0,1]
    img_float = img.astype(np.float32) / 255.0
    # Estimate the noise standard deviation
    sigma_est = np.mean(estimate_sigma(img_float, multichannel=True))
    # Apply BM3D denoising
    denoised_img = bm3d(img_float, sigma_psd=sigma_est)
    # Convert back to uint8
    denoised_img = np.clip(denoised_img * 255.0, 0, 255).astype(np.uint8)
    return denoised_img

# Function to denoise image using DnCNN
def denoise_image_with_dncnn(img):
    # Convert image to tensor
    img_float = img.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_float.transpose(2, 0, 1)).unsqueeze(0)
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_tensor = img_tensor.to(device)
    # Load pre-trained DnCNN model
    model = DnCNN()
    # Ensure the model weights are available
    if not os.path.exists('dncnn.pth'):
        raise FileNotFoundError("Pre-trained DnCNN model weights 'dncnn.pth' not found.")
    model.load_state_dict(torch.load('dncnn.pth', map_location=device))
    model.to(device)
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
    denoised_img = output.squeeze(0).cpu().permute(1, 2, 0).numpy()
    denoised_img = np.clip(denoised_img * 255.0, 0, 255).astype(np.uint8)
    return denoised_img

# Function to denoise image using Wiener filter
def denoise_image_with_wiener(img):
    # Convert to float
    img_float = img.astype(np.float32) / 255.0
    # Apply Wiener filter
    denoised_img = np.zeros_like(img_float)
    for i in range(3):  # Process each channel separately
        denoised_img[:, :, i] = wiener(img_float[:, :, i])
    # Convert back to uint8
    denoised_img = np.clip(denoised_img * 255.0, 0, 255).astype(np.uint8)
    return denoised_img

def resize_image(img, target_width=1440):
    aspect_ratio = img.shape[0] / img.shape[1]
    new_height = int(target_width * aspect_ratio)
    return cv2.resize(img, (target_width, new_height))

def clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe_obj = cv2.createCLAHE(clipLimit=0.8, tileGridSize=(8, 8))
    l_clahe = clahe_obj.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

def gamma_correction_light(image, gamma=1.4):
    img_np = np.array(image)
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** 1.2 * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img_np, table)

def cut(img):
    edge = cv2.Canny(img, 50, 150)
    texture_mask = cv2.dilate(edge, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    contours, _ = cv2.findContours(texture_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cropped_img = img[y:y + h, x:x + w]
        return cropped_img
    return img  # Return the original image if no contours are found

# Function to detect the type and level of noise in the image
def detect_noise_type_and_level(img_gray):
    # Compute the Laplacian of the image and its variance
    laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
    laplacian_var = laplacian.var()

    # Compute the variance of the image
    img_var = img_gray.var()

    # Estimate noise level
    noise_level = np.mean(cv2.absdiff(img_gray, cv2.GaussianBlur(img_gray, (3, 3), 0)))

    # Thresholds are empirically determined
    if laplacian_var < 10 and img_var < 100:
        noise_type = 'gaussian'
    elif laplacian_var > 1000:
        noise_type = 'salt_pepper'
    else:
        noise_type = 'none'

    return noise_type, noise_level

# Function to sharpen the image
def sharpen_image(img):
    # Create a Gaussian blurred image
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=3)
    # Subtract the blurred image from the original image
    sharpened = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
    return sharpened

def process_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        logging.error(f"Failed to read image {img_path}")
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized_img = resize_image(img_rgb, target_width=1440)

    # Detect noise type and level
    img_gray = cv2.cvtColor(resized_img, cv2.COLOR_RGB2GRAY)
    noise_type, noise_level = detect_noise_type_and_level(img_gray)
    logging.info(f"{os.path.basename(img_path)} - Detected noise type: {noise_type}, Noise level: {noise_level:.2f}")

    # Adaptive denoising based on noise level
    if noise_level < 5:
        # Minimal denoising using fastNlMeansDenoisingColored
        h_value = 3
        denoised_img = cv2.fastNlMeansDenoisingColored(resized_img, None, h=h_value, hColor=h_value,
                                                       templateWindowSize=7, searchWindowSize=21)
    elif noise_level < 10:
        # Moderate denoising using Wiener filter
        denoised_img = denoise_image_with_wiener(resized_img)
    elif noise_level < 15:
        # Stronger denoising using BM3D
        denoised_img = denoise_image_with_bm3d(resized_img)
    else:
        # Strong denoising using DnCNN
        try:
            denoised_img = denoise_image_with_dncnn(resized_img)
        except Exception as e:
            logging.error(f"Error using DnCNN: {e}")
            logging.info("Falling back to BM3D denoising")
            denoised_img = denoise_image_with_bm3d(resized_img)
        if noise_type == 'salt_pepper':
            denoised_img = cv2.medianBlur(denoised_img, 3)

    # Compute PSNR and SSIM between denoised image and original image
    psnr_value = compare_psnr(resized_img, denoised_img, data_range=255)

    # Determine the win_size based on image dimensions
    image_height, image_width = resized_img.shape[:2]
    min_dimension = min(image_height, image_width)

    if min_dimension < 7:
        win_size = min_dimension if min_dimension % 2 == 1 else min_dimension - 1
        if win_size < 3:
            win_size = 3  # Set a minimum win_size of 3
    else:
        win_size = 7  # Use the default win_size

    try:
        ssim_value = compare_ssim(
            resized_img,
            denoised_img,
            data_range=255,
            channel_axis=-1,
            win_size=win_size
        )
    except ValueError as e:
        logging.warning(f"{os.path.basename(img_path)} - SSIM calculation error: {e}")
        ssim_value = 1.0  # Assign a default SSIM value

    logging.info(f"{os.path.basename(img_path)} - PSNR after denoising: {psnr_value:.2f}, SSIM: {ssim_value:.4f}")

    # Adaptive sharpening based on SSIM
    if ssim_value < 0.8:
        sharpened_img = sharpen_image(denoised_img)
    else:
        sharpened_img = denoised_img

    # Proceed with the rest of the processing
    cropped_img = cut(sharpened_img)
    enhanced_img = clahe(cropped_img)
    gamma_img = gamma_correction_light(enhanced_img)

    return gamma_img


# Function to save the image using Pillow for better JPEG compression
def save_image_with_pillow(image, output_path, jpeg_quality):
    pil_image = Image.fromarray(image)  # Assuming image is in RGB format
    pil_image.save(output_path, "JPEG", quality=jpeg_quality, optimize=True)

# Function to process a single image
def process_single_image(img_path, output_folder, jpeg_quality=95):
    filename = os.path.basename(img_path)
    try:
        enhanced_image = process_image(img_path)
        if enhanced_image is not None:
            # Output path
            output_path = os.path.join(output_folder, filename)
            save_image_with_pillow(enhanced_image, output_path, jpeg_quality)
            logging.info(f"Saved enhanced image {output_path}")
        else:
            logging.error(f"Failed to process image {img_path}")
    except Exception as e:
        logging.error(f"Error processing {filename}: {str(e)}")

# Function to process images in parallel
def process_images_in_parallel(input_folder, output_folder, jpeg_quality=95, max_workers=4):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_paths = [
        os.path.join(input_folder, filename) for filename in os.listdir(input_folder)
        if filename.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_image, img_path, output_folder, jpeg_quality)
                   for img_path in image_paths]

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logging.error(f"Error: {str(e)}")

# Main execution block
if __name__ == "__main__":
    input_folder = 'D:\\Desktop\\AI361\\project\\dates-contest-images'
    output_folder = 'D:\\Desktop\\AI361\\project\\enhanced-images29'
    jpeg_quality = 97
    max_workers = 4

    process_images_in_parallel(input_folder, output_folder, jpeg_quality, max_workers)
