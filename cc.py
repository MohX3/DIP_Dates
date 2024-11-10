import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor


# Function to apply denoising, contrast, and sharpening enhancements
def denoise_and_enhance(image):
    """Denoises the image, enhances contrast, and sharpens it."""
    # Step 1: Apply Non-Local Means Denoising
    denoised_image = cv2.fastNlMeansDenoising(image, None, h=10, templateWindowSize=7, searchWindowSize=21)

    # Step 2: Enhance contrast using CLAHE with lower clip limit
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    contrast_enhanced_image = clahe.apply(denoised_image)

    #Step 3: Sharpen using unsharp mask
    sharpened_image = unsharp_mask(contrast_enhanced_image)

    return sharpened_image


# Function to apply unsharp mask for sharpening
def unsharp_mask(image, sigma=1.0, strength=1.5):
    """Applies unsharp mask to enhance image sharpness."""
    blur = cv2.GaussianBlur(image, (0, 0), sigma)
    sharpened = cv2.addWeighted(image, 1.0 + strength, blur, -strength, 0)
    return sharpened





# Function to resize image to reduce storage space
def resize_image(image, width):
    aspect_ratio = image.shape[1] / image.shape[0]  # Calculate aspect ratio
    new_height = int(width / aspect_ratio)  # Calculate new height to maintain aspect ratio
    resized_image = cv2.resize(image, (width, new_height), interpolation=cv2.INTER_LINEAR)
    return resized_image


# Single image processing function for parallel execution
def process_single_image(filename):
    img_path = os.path.join(input_folder, filename)
    gray_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if gray_image is None:
        print(f"Error reading image: {filename}")
        return

    # Step 1: Resize the image
    resized_image = resize_image(gray_image, resize_width)

    # Step 2: Denoise and enhance the image
    enhanced_image = denoise_and_enhance(resized_image)

    # Step 3: Save the processed image
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, enhanced_image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    print(f"Saved processed image: {output_path}")


# Process all images in parallel using ThreadPoolExecutor
def process_images_with_enhancement_parallel(input_folder, output_folder):
    """Processes images in parallel: denoising, enhancing, resizing, and saving."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    images = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(process_single_image, images)


# Example usage
input_folder = 'C:\\Users\\96658\\Desktop\\AI361\\project\\dates-contest-images'
output_folder = 'C:\\Users\\96658\\Desktop\\AI361\\project\\enhanced-images3'
resize_width = 500
jpeg_quality = 90

process_images_with_enhancement_parallel(input_folder, output_folder)
