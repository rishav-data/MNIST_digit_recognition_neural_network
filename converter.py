import cv2
import numpy as np
import csv
import os

def convert_image():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, "static", "uploads", "digit.png")
    csv_path = os.path.join(script_dir, "input.csv")

    print("🔍 Looking for image at:", image_path)

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ Image not found at: {image_path}")

    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("❌ Failed to read image — file may be corrupted or not an image.")

    # Invert if background is white
    if np.mean(img) > 127:
        img = 255 - img

    # Threshold to remove noise
    _, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

    # Find bounding box of digit
    coords = cv2.findNonZero(img)
    x, y, w, h = cv2.boundingRect(coords)

    # Crop around digit
    cropped = img[y:y+h, x:x+w]

    # Resize cropped digit to 20x20
    resized = cv2.resize(cropped, (20, 20), interpolation=cv2.INTER_AREA)

    # Pad to 28x28 (centered)
    padded = np.pad(resized, ((4, 4), (4, 4)), mode='constant', constant_values=0)

    # Normalize to 0–1
    img_normalized = padded / 255.0

    # Flatten and save to CSV
    flat_img = img_normalized.flatten()
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(flat_img)

    print("✅ Image processed and saved as:", csv_path)

if __name__ == "__main__":
    convert_image()
