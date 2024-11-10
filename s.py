import cv2
import numpy as np


def apply_mask(img, mask):
    white_background = np.full_like(img, 255)
    return np.where(mask[:, :, None] == 0, img, white_background)


def segment_date(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, binary_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # adaptive = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    return binary_mask


def resize_image(img, target_width=500):
    aspect_ratio = img.shape[1] / img.shape[0]
    new_height = int(target_width / aspect_ratio)
    return cv2.resize(img, (target_width, new_height))


def clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
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


def process_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized_img = resize_image(img, target_width=1000)

    # mask = segment_date(resized_img)
    # masked_img = apply_mask(resized_img, mask)

    cropped_img = cut(resized_img)
    median_img = cv2.bilateralFilter(cropped_img, 3, 3, 10)
    enhanced_img = clahe(median_img)
    # log_img = log_transform(cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2GRAY))
    sharpened_img = cv2.pyrUp(enhanced_img)
    gamma_img = gamma_correction_light(sharpened_img)

    return gamma_img