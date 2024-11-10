import cv2
import numpy as np
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)


def process_image(input_path, output_path, debug=False):
    # Read the image in grayscale
    image_gray = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
    if image_gray is None:
        logging.warning(f"Failed to load image {input_path}. Skipping.")
        return

    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image_gray)

    # Apply Non-Local Means Denoising to reduce noise
    denoised = cv2.fastNlMeansDenoising(enhanced, None, h=15, templateWindowSize=7, searchWindowSize=21)

    # Adaptive thresholding to highlight object
    blockSize = 51  # Adjust block size for larger objects
    C = 2  # Threshold adjustment constant
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize, C
    )

    # Morphological operations to fill holes and remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

    # Detect edges and combine with the thresholded image
    edges = cv2.Canny(opened, 50, 150)
    combined = cv2.bitwise_or(opened, edges)

    # Find contours on the combined image
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Calculate image center
        height, width = image_gray.shape
        image_center = (width // 2, height // 2)

        # Filter and find the contour closest to the center
        min_area = 0.05 * height * width  # Minimum contour area (5% of image area)
        suitable_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

        if suitable_contours:
            closest_contour = min(
                suitable_contours,
                key=lambda cnt: abs(cv2.pointPolygonTest(cnt, image_center, True))
            )

            # Use convex hull to better fit the object and draw contours
            hull = cv2.convexHull(closest_contour)
            epsilon = 0.01 * cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, epsilon, True)
            cv2.drawContours(image_gray, [approx], -1, (0, 255, 0), 2)  # Green contour

            # Create a mask for the contour and dilate it
            contour_mask = np.zeros_like(image_gray)
            cv2.drawContours(contour_mask, [approx], -1, 255, cv2.FILLED)
            dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
            dilated_mask = cv2.dilate(contour_mask, dilate_kernel, iterations=1)

            # Find bounding box for the dilated mask
            dilated_contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if dilated_contours:
                x, y, w, h = cv2.boundingRect(dilated_contours[0])
                # Draw bounding box with padding around the object
                cv2.rectangle(image_gray, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue box
                cv2.imwrite(str(output_path), image_gray)
                logging.info(f"Saved image with bounding box to {output_path}")
            else:
                logging.warning(f"No contours found after dilation in {input_path}")
        else:
            logging.warning(f"No suitable contours found in {input_path}")
            # Use a fallback bounding box (almost entire image)
            fallback_x, fallback_y = int(0.05 * width), int(0.05 * height)
            fallback_w, fallback_h = int(0.9 * width), int(0.9 * height)
            cv2.rectangle(image_gray, (fallback_x, fallback_y), (fallback_x + fallback_w, fallback_y + fallback_h),
                          (255, 0, 0), 2)  # Fallback box
            cv2.imwrite(str(output_path), image_gray)
            logging.info(f"Used fallback bounding box for {output_path}")
    else:
        logging.warning(f"No contours detected in {input_path}")

    # Optional debug images
    if debug:
        debug_folder = output_path.parent / 'debug'
        debug_folder.mkdir(exist_ok=True, parents=True)
        cv2.imwrite(str(debug_folder / f"{input_path.stem}_enhanced.jpg"), enhanced)
        cv2.imwrite(str(debug_folder / f"{input_path.stem}_denoised.jpg"), denoised)
        cv2.imwrite(str(debug_folder / f"{input_path.stem}_thresh.jpg"), thresh)
        cv2.imwrite(str(debug_folder / f"{input_path.stem}_combined.jpg"), combined)


# Paths
input_folder = Path('D:/Desktop/AI361/project/enhanced-images13')
output_folder = Path('D:/Desktop/AI361/project/output-images')
output_folder.mkdir(parents=True, exist_ok=True)

# Process each image
for filename in input_folder.iterdir():
    if filename.suffix.lower() in ('.jpg', '.jpeg', '.png'):
        input_path = filename
        output_path = output_folder / filename.name
        process_image(input_path, output_path, debug=True)
