import os
import cv2
import numpy as np
from glob import glob

# Augmentation functions

def rotate(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))

def zoom(image, zoom_factor):
    h, w = image.shape[:2]
    new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
    image = cv2.resize(image, (new_w, new_h))
    if zoom_factor < 1:
        pad_h = (h - new_h) // 2
        pad_w = (w - new_w) // 2
        image = cv2.copyMakeBorder(image, pad_h, h - new_h - pad_h, pad_w, w - new_w - pad_w, cv2.BORDER_CONSTANT)
    else:
        image = image[:h, :w]
    return image

def adjust_brightness(image, value):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:,:,2] = np.clip(hsv[:,:,2] + value, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def flip(image, mode):
    return cv2.flip(image, mode)

# Main augmentation pipeline

def augment_images(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    image_paths = glob(os.path.join(input_dir, '*.jpg'))
    for img_path in image_paths:
        img = cv2.imread(img_path)
        basename = os.path.splitext(os.path.basename(img_path))[0]
        # Rotation
        for angle in [15, -15]:
            aug = rotate(img, angle)
            cv2.imwrite(os.path.join(output_dir, f'{basename}_rot{angle}.jpg'), aug)
        # Zoom
        for zf in [0.9, 1.1]:
            aug = zoom(img, zf)
            cv2.imwrite(os.path.join(output_dir, f'{basename}_zoom{zf}.jpg'), aug)
        # Brightness
        for val in [30, -30]:
            aug = adjust_brightness(img, val)
            cv2.imwrite(os.path.join(output_dir, f'{basename}_bright{val}.jpg'), aug)
        # Flip
        for mode in [0, 1]:
            aug = flip(img, mode)
            cv2.imwrite(os.path.join(output_dir, f'{basename}_flip{mode}.jpg'), aug)

if __name__ == "__main__":
    augment_images('input_images', 'augmented_images')
