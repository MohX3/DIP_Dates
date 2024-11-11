import os
import cv2
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed


# Function for CLAHE and image enhancement
def enhance_image(image):
    # Convert image to LAB color space for better luminance processing
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Apply CLAHE (Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # Adjusted parameters for better enhancement
    cl = clahe.apply(l_channel)

    # Merge the CLAHE enhanced L-channel back with A and B channels
    limg = cv2.merge((cl, a_channel, b_channel))

    # Convert image back to BGR color space
    enhanced_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # Apply Gaussian Blur
    blurred_image = cv2.GaussianBlur(enhanced_image, (3, 3), 0)

    # Apply selective denoising
    denoised_image = selective_denoising(blurred_image)

    # Enhanced sharpening with Unsharp Mask
    sharpened_image = cv2.addWeighted(denoised_image, 1.5, cv2.GaussianBlur(denoised_image, (0, 0), 3), -0.5, 0)

    return sharpened_image


# Selective denoising function
def selective_denoising(image):
    # Convert image to grayscale for edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use Laplacian to detect edges
    edges = cv2.Laplacian(gray, cv2.CV_64F)
    edges = np.absolute(edges)
    edges = np.uint8(edges)

    # Apply stronger denoising to non-edge areas
    non_edge_mask = edges < 30  # Areas with less edge intensity
    denoised_image = cv2.fastNlMeansDenoisingColored(
        image, None, h=3, hColor=3, templateWindowSize=7, searchWindowSize=21
    )

    # Combine denoised image with original using the mask
    image[non_edge_mask] = denoised_image[non_edge_mask]

    return image


# Function to resize image using Lanczos interpolation
def resize_image_lanczos(image, width):
    aspect_ratio = image.shape[1] / image.shape[0]
    new_height = int(width / aspect_ratio)
    resized_image = cv2.resize(image, (width, new_height), interpolation=cv2.INTER_LANCZOS4)
    return resized_image


# Function to save the image using Pillow for better JPEG compression
def save_image_with_pillow(image, output_path, jpeg_quality):
    # Convert OpenCV image (BGR) to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert to PIL Image
    pil_image = Image.fromarray(image_rgb)

    # Save using Pillow
    pil_image.save(output_path, "JPEG", quality=jpeg_quality, optimize=True)


# Function to process a single image
def process_single_image(img_path, output_folder, resize_width=None, jpeg_quality= None):
    filename = os.path.basename(img_path)
    try:
        image = cv2.imread(img_path)

        if image is not None:
            # Resize the image first
            if resize_width is not None:
                resized_image = resize_image_lanczos(image, resize_width)
            else:
                resized_image = image

            # Apply enhancements after resizing
            enhanced_image = enhance_image(resized_image)

            # Create output path
            output_path = os.path.join(output_folder, filename)

            # Save using Pillow for better compression
            save_image_with_pillow(enhanced_image, output_path, jpeg_quality)

            print(f"Saved enhanced image {output_path}")
        else:
            print(f"Failed to load image {img_path}")
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")


# Function to process all images in parallel
def process_images_in_parallel(input_folder, output_folder, resize_width=None, jpeg_quality=98, max_workers=4):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get the list of image paths
    image_paths = [
        os.path.join(input_folder, filename) for filename in os.listdir(input_folder)
        if filename.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    # Use ThreadPoolExecutor to process images in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_image, img_path, output_folder, resize_width, jpeg_quality)
                   for img_path in image_paths]

        for future in as_completed(futures):
            try:
                future.result()  # Retrieve the result (if any) to raise any exceptions that occurred
            except Exception as e:
                print(f"Error: {str(e)}")


# Example usage
input_folder = 'D:\\Desktop\\AI361\\project\\dates-contest-images'
output_folder = 'D:\\Desktop\\AI361\\project\\enhanced-images16'
resize_width = 1800  # Resize width, adjust as necessary
jpeg_quality = 80  # Max JPEG quality to maintain under the 80 MB file size limit
max_workers = 4  # Number of workers for parallel processing

process_images_in_parallel(input_folder, output_folder, resize_width, jpeg_quality, max_workers)
