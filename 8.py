import os
import cv2
import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed


def apply_mask(img, mask):
    white_background = np.full_like(img, 255)
    return np.where(mask[:, :, None] == 0, img, white_background)


def segment_date(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, binary_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # adaptive = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    return binary_mask


def resize_image(img, target_width=1440):
    aspect_ratio = img.shape[1] / img.shape[0]
    new_height = int(target_width / aspect_ratio)
    return cv2.resize(img, (target_width, new_height))


def clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=0.8, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)


def log_transform(image):
    img_np = image / 255.0
    log_transformed = 2 * (np.log(1 + img_np))
    return np.uint8(255 * log_transformed / np.max(log_transformed))


def selective_sharpening(img, mask=None):
    kernel_sharpening = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened_img = cv2.filter2D(img, -1, kernel_sharpening * 0.4)
    if mask is not None:
        result_img = cv2.bitwise_and(sharpened_img, sharpened_img, mask=mask)
        result_img += cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask))
    else:
        result_img = sharpened_img
    return result_img


def gamma_correction_light(image, gamma=1.4):
    img_np = np.array(image)
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img_np, table)


def crop_to_date(img, mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cropped_img = img[y:y + h, x:x + w]
        return cropped_img
    return img


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


def process_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized_img = resize_image(img, target_width=1440)

    # Uncomment the following lines if you want to apply masking
    # mask = segment_date(resized_img)
    # masked_img = apply_mask(resized_img, mask)

    cropped_img = cut(resized_img)
    median_img = cv2.bilateralFilter(cropped_img, 3, 3, 10)
    enhanced_img = clahe(median_img)
    # Uncomment the following line if you want to apply log transformation
    # log_img = log_transform(cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2GRAY))
    sharpened_img = cv2.pyrUp(enhanced_img)
    gamma_img = gamma_correction_light(sharpened_img)

    return gamma_img


# Function 7: Save the image using Pillow for better JPEG compression
def save_image_with_pillow(image, output_path, jpeg_quality):
    pil_image = Image.fromarray(image)  # Assuming image is in RGB format
    pil_image.save(output_path, "JPEG", quality=jpeg_quality, optimize=True)


# Function 8: Process a single image
def process_single_image(img_path, output_folder, jpeg_quality=80):
    filename = os.path.basename(img_path)
    try:
        enhanced_image = process_image(img_path)
        if enhanced_image is not None:
            # Output path
            output_path = os.path.join(output_folder, filename)
            save_image_with_pillow(enhanced_image, output_path, jpeg_quality)
            print(f"Saved enhanced image {output_path}")
        else:
            print(f"Failed to process image {img_path}")
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")


# Function 9: Process images in parallel
def process_images_in_parallel(input_folder, output_folder, jpeg_quality=80, max_workers=4):
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
                print(f"Error: {str(e)}")


# Function 10: Main execution block
if __name__ == "__main__":
    input_folder = 'D:\\Desktop\\AI361\\project\\dates-contest-images'
    output_folder = 'D:\\Desktop\\AI361\\project\\enhanced-images221'
    jpeg_quality = 90
    max_workers = 4

    process_images_in_parallel(input_folder, output_folder, jpeg_quality, max_workers)
