import os
import cv2
import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
import imghdr
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)

# Function for CLAHE and image enhancement
def enhance_image(image, clahe_clip_limit=2.0, clahe_tile_grid_size=(8, 8)):
    # Convert image to LAB color space for better luminance processing
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Apply CLAHE (Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid_size)
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

    # Use Canny edge detection
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)

    # Apply stronger denoising to non-edge areas
    non_edge_mask = edges == 0  # Non-edge areas

    denoised_image = cv2.fastNlMeansDenoisingColored(
        image, None, h=3, hColor=3, templateWindowSize=7, searchWindowSize=21
    )

    # Create a copy of the image to avoid modifying the original
    output_image = image.copy()
    output_image[non_edge_mask] = denoised_image[non_edge_mask]

    return output_image

# Function to resize image using high-quality interpolation
def resize_image_high_quality(image, width):
    aspect_ratio = image.shape[1] / image.shape[0]
    new_height = int(width / aspect_ratio)

    # Determine if scaling up or down for best interpolation method
    if width > image.shape[1]:
        # Scaling up, use Lanczos interpolation
        interpolation = cv2.INTER_LANCZOS4
    else:
        # Scaling down, use Area interpolation
        interpolation = cv2.INTER_AREA

    resized_image = cv2.resize(image, (width, new_height), interpolation=interpolation)
    return resized_image

# Function to save the image using Pillow with optimized compression
def save_image_with_pillow(image, output_path, jpeg_quality):
    # Convert OpenCV image (BGR) to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert to PIL Image
    pil_image = Image.fromarray(image_rgb)

    # Save using Pillow with optimized settings
    pil_image.save(
        output_path,
        "JPEG",
        quality=jpeg_quality,
        optimize=True,
        progressive=True,
        subsampling=0  # Set subsampling to 4:4:4 for higher quality
    )

# Function to process a single image
def process_single_image(img_path, output_folder, resize_width=None, jpeg_quality=90, clahe_clip_limit=2.0, clahe_tile_grid_size=(8, 8)):
    filename = os.path.basename(img_path)
    try:
        image = cv2.imread(img_path)

        if image is not None:
            # Resize the image first
            if resize_width is not None:
                resized_image = resize_image_high_quality(image, resize_width)
            else:
                resized_image = image

            # Apply enhancements after resizing
            enhanced_image = enhance_image(resized_image, clahe_clip_limit, clahe_tile_grid_size)

            # Create output path
            output_path = os.path.join(output_folder, filename)

            # Save using Pillow for better compression
            save_image_with_pillow(enhanced_image, output_path, jpeg_quality)

            logging.info(f"Saved enhanced image {output_path}")
        else:
            logging.error(f"Failed to load image {img_path}")
    except Exception as e:
        logging.error(f"Error processing {filename}: {str(e)}")
        traceback.print_exc()

# Function to process all images in parallel
def process_images_in_parallel(input_folder, output_folder, resize_width=None, jpeg_quality=90, max_workers=4, clahe_clip_limit=2.0, clahe_tile_grid_size=(8, 8)):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get the list of image paths
    image_paths = [
        os.path.join(input_folder, filename) for filename in os.listdir(input_folder)
        if imghdr.what(os.path.join(input_folder, filename)) is not None
    ]

    # Use ProcessPoolExecutor to process images in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_image, img_path, output_folder, resize_width, jpeg_quality, clahe_clip_limit, clahe_tile_grid_size)
                   for img_path in image_paths]

        for future in as_completed(futures):
            try:
                future.result()  # Retrieve the result to raise any exceptions that occurred
            except Exception as e:
                logging.error(f"Error: {str(e)}")
                traceback.print_exc()

# Main function that can be called directly
def main():
    # Set your parameters here
    input_folder = 'C:\\Users\\96658\\Desktop\\AI361\\project\\dates-contest-images'
    output_folder = 'C:\\Users\\96658\\Desktop\\AI361\\project\\enhanced-images14'
    resize_width = 1600  # Reduce this value slightly if needed
    jpeg_quality = 95  # Lowered from 95 to 90 to reduce file size
    max_workers = 4
    clahe_clip_limit = 2.0
    clahe_tile_grid_size = (8, 8)

    process_images_in_parallel(
        input_folder,
        output_folder,
        resize_width,
        jpeg_quality,
        max_workers,
        clahe_clip_limit,
        clahe_tile_grid_size
    )

if __name__ == "__main__":
    main()
