import cv2
import numpy as np
import os

def process_image(image_path, output_path):
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image '{image_path}'. Please check the file path or format.")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Reduce shadows (apply morphological transformation)
    dilated_img = cv2.dilate(gray, np.ones((15, 15), np.uint8))
    bg_removed = cv2.divide(gray, dilated_img, scale=255)

    # Apply High-Pass Filter (Fourier Transform)
    f = np.fft.fft2(bg_removed)
    fshift = np.fft.fftshift(f)
    rows, cols = bg_removed.shape
    crow, ccol = rows // 2, cols // 2
    # Suppress low frequencies (center area)
    mask = np.ones((rows, cols), np.uint8)
    mask[crow-30:crow+30, ccol-30:ccol+30] = 0
    fshift *= mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    img_back = np.uint8(img_back)

    # Edge Detection
    edges = cv2.Canny(img_back, 50, 150)

    # Find Contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by area to get the main object
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1000]
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Draw bounding box with padding
        padding = 10
        x, y = max(0, x - padding), max(0, y - padding)
        w, h = min(image.shape[1] - x, w + 2 * padding), min(image.shape[0] - y, h + 2 * padding)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Save result
        cv2.imwrite(output_path, image)
        print(f"Processed image saved at {output_path}")
    else:
        print("No valid contour found.")

# Example usage
input_folder = 'D:\\Desktop\\AI361\\project\\dates-contest-images'
output_folder = 'D:\\Desktop\\AI361\\project\\cropped-images'

# Process images from 1 to 600
for i in range(1, 601):
    input_path = os.path.join(input_folder, f"{i}.jpg")
    output_path = os.path.join(output_folder, f"processed_{i}.jpg")
    process_image(input_path, output_path)
