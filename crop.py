import cv2
import numpy as np
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)

def process_image(input_path, output_path, debug=False):
    # Read the image in color
    image_color = cv2.imread(str(input_path))
    if image_color is None:
        logging.warning(f"Failed to load image {input_path}. Skipping.")
        return

    # Convert to grayscale
    image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image_gray)

    # Apply Non-Local Means Denoising
    denoised = cv2.fastNlMeansDenoising(enhanced, None, h=10, templateWindowSize=7, searchWindowSize=21)

    # Apply adaptive thresholding
    blockSize = 31  # Must be odd and greater than 1
    C = 2  # Adjust as needed
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize, C
    )

    # Apply morphological operations to reduce noise and refine the object
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # First, perform closing to fill small holes
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    # Then, perform opening to remove small noise
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=2)

    # Use edge detection to better define object boundaries
    edges = cv2.Canny(opened, 50, 150)

    # Combine edges with the opened image
    combined = cv2.bitwise_or(opened, edges)

    # Find contours on the combined image
    contours, hierarchy = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Image area
        height, width = image_gray.shape
        image_area = height * width
        min_area = image_area * 0.01  # Adjust as needed

        # Filter contours based on area
        possible_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]

        if possible_contours:
            # Get the largest contour
            largest_contour = max(possible_contours, key=cv2.contourArea)

            # Use convex hull to better fit the object
            hull = cv2.convexHull(largest_contour)

            # Optionally, approximate the contour to simplify it
            epsilon = 0.01 * cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, epsilon, True)

            # Draw the original contour on the image
            cv2.drawContours(image_color, [approx], -1, (0, 255, 0), 2)  # Green contour

            # Create a mask of the contour
            contour_mask = np.zeros_like(image_gray)
            cv2.drawContours(contour_mask, [approx], -1, 255, cv2.FILLED)

            # Dilate the contour to expand it by 8 pixels
            padding_pixels = 8  # Desired gap in pixels
            dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (padding_pixels * 2, padding_pixels * 2))
            dilated_mask = cv2.dilate(contour_mask, dilate_kernel, iterations=1)

            # Find contours on the dilated mask
            dilated_contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if dilated_contours:
                # Get the bounding rectangle of the dilated contour
                x_dilated, y_dilated, w_dilated, h_dilated = cv2.boundingRect(dilated_contours[0])

                # Ensure coordinates are within image bounds
                x_box = max(0, x_dilated)
                y_box = max(0, y_dilated)
                x_box_end = min(image_color.shape[1], x_dilated + w_dilated)
                y_box_end = min(image_color.shape[0], y_dilated + h_dilated)

                # Draw the bounding box with an 8-pixel gap around the object
                cv2.rectangle(image_color, (x_box, y_box), (x_box_end, y_box_end), (255, 0, 0), 2)  # Blue box

                # Save the original image with the bounding box
                cv2.imwrite(str(output_path), image_color)
                logging.info(f"Saved image with bounding box to {output_path}")
            else:
                logging.warning(f"No contours found after dilation in {input_path}")
        else:
            logging.warning(f"No suitable contours found in {input_path}")
    else:
        logging.warning(f"No contours detected in {input_path}")

    # Optional: Save debug images
    if debug:
        debug_folder = output_path.parent / 'debug'
        debug_folder.mkdir(exist_ok=True)
        cv2.imwrite(str(debug_folder / f"{input_path.stem}_enhanced.jpg"), enhanced)
        cv2.imwrite(str(debug_folder / f"{input_path.stem}_denoised.jpg"), denoised)
        cv2.imwrite(str(debug_folder / f"{input_path.stem}_thresh.jpg"), thresh)
        cv2.imwrite(str(debug_folder / f"{input_path.stem}_closed.jpg"), closed)
        cv2.imwrite(str(debug_folder / f"{input_path.stem}_opened.jpg"), opened)
        cv2.imwrite(str(debug_folder / f"{input_path.stem}_edges.jpg"), edges)
        cv2.imwrite(str(debug_folder / f"{input_path.stem}_combined.jpg"), combined)
        cv2.imwrite(str(debug_folder / f"{input_path.stem}_contour_mask.jpg"), contour_mask)
        cv2.imwrite(str(debug_folder / f"{input_path.stem}_dilated_mask.jpg"), dilated_mask)
        # Draw all contours for debugging
        debug_image = image_color.copy()
        cv2.drawContours(debug_image, contours, -1, (0, 255, 0), 1)
        cv2.imwrite(str(debug_folder / f"{input_path.stem}_contours.jpg"), debug_image)

# Files
input_folder = Path('D:/Desktop/AI361/project/enhanced-images13')
output_folder = Path('D:/Desktop/AI361/project/output-images')

# Ensure output folder exists
output_folder.mkdir(parents=True, exist_ok=True)

# Process each image in the input folder
for filename in input_folder.iterdir():
    if filename.suffix.lower() in ('.jpg', '.jpeg', '.png'):
        input_path = filename
        output_path = output_folder / filename.name
        process_image(input_path, output_path, debug=True)
