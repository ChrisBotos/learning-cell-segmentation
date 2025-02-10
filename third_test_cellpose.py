import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from skimage import io as skio
from skimage.measure import regionprops_table, label
from cellpose import models, plot

# === SETTINGS ===
UPSCALE_FACTOR = 1  # Set >1 to upscale (e.g., 2 for 2× zoom), or 1 for no upscaling
CROP_IMAGE = True
enhance_gray_parts = False

# === SETUP OUTPUT DIRECTORY ===
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# === GPU CHECK ===
print("=== Checking GPU Availability ===")
torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {torch_device}")
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))

# === LOAD IMAGE ===
image_path = "DAPI_bIRI2.tif"
image = skio.imread(image_path)
print(f"Original Image: dtype={image.dtype}, shape={image.shape}")

# === HANDLE RGBA IMAGES ===
if image.ndim == 3 and image.shape[-1] == 4:
    image = image[:, :, :3]  # Remove alpha channel
    print("Removed alpha channel.")

# === CONVERT 16-BIT TO 8-BIT ===
if image.dtype == np.uint16:
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    print("Converted 16-bit image to 8-bit.")

# === CONVERT TO GRAYSCALE ===
if image.ndim == 3:
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    print("Converted image to grayscale.")

# === CROP IMAGE ===
if CROP_IMAGE:
    h, w = image.shape
    image = image[h // 4: h // 3, w // 3 : w // 2]
    print(f"Cropped Image Shape: {image.shape}")


# === UPSCALE IMAGE ===
if UPSCALE_FACTOR > 1:
    image = cv2.resize(image, None, fx=UPSCALE_FACTOR, fy=UPSCALE_FACTOR, interpolation=cv2.INTER_LINEAR)
    print(f"Upscaled Image Shape: {image.shape}")

# === SAVE PROCESSED IMAGE ===
gray_image_path = os.path.join(output_dir, "processed_image.png")
skio.imsave(gray_image_path, image)
print(f"Saved processed grayscale image: {gray_image_path}")

# === PRINT IMAGE STATS ===
print(f"Image min: {image.min()}, max: {image.max()}, mean: {image.mean()}")

# === CONTRAST ENHANCEMENT (CLAHE) ===
clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(16, 16))
image = clahe.apply(image)

# === ENHANCE DIM REGIONS ===
if enhance_gray_parts:
    # Estimate the background using a large kernel
    background = cv2.GaussianBlur(image, (101, 101), 0)

    # Subtract the background from the original image
    foreground = cv2.subtract(image, background)

    # Normalize the foreground to enhance contrast
    foreground = cv2.normalize(foreground, None, 0, 255, cv2.NORM_MINMAX)

    # Save and display the foreground image
    skio.imsave(os.path.join(output_dir, "foreground_image.png"), foreground)
    plt.imshow(foreground, cmap="gray")
    plt.title("Foreground Image (Background Subtracted)")
    plt.axis("off")
    plt.show()

else:
    foreground = image

# === RUN CELLPOSE SEGMENTATION ===
model = models.Cellpose(model_type='nuclei', gpu=torch.cuda.is_available())

masks, flows, styles, diams = model.eval(
    foreground,
    diameter=5 * UPSCALE_FACTOR,  # Adjust diameter if upscaled.
    channels=[0, 0],
    flow_threshold=0.5,
    cellprob_threshold=-2,
    resample=True
)

# === SAVE MASK IMAGE ===
mask_path = os.path.join(output_dir, "segmentation_mask.png")
skio.imsave(mask_path, masks.astype(np.uint16))  # Save as 16-bit mask

# === NORMALIZE & SAVE MASK FOR VISUALIZATION ===
normalized_mask = (masks / masks.max() * 255).astype(np.uint8) if masks.max() > 0 else masks
cv2.imwrite(os.path.join(output_dir, "segmentation_mask_visual.png"), normalized_mask)

# === CREATE & SAVE MASK OVERLAY ===
mask_overlay = plot.mask_overlay(image, masks, colors=np.random.rand(np.max(masks) + 1, 3))
overlay_path = os.path.join(output_dir, "mask_overlay.png")
skio.imsave(overlay_path, (mask_overlay * 255).astype(np.uint8))
print(f"Saved mask overlay: {overlay_path}")

# === DISPLAY RESULTS ===
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap="gray")
plt.title("Processed DAPI Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(mask_overlay)
plt.title("Cellpose Segmentation Mask")
plt.axis("off")

plt.savefig(os.path.join(output_dir, "mask_overlay_plot.png"), dpi=300)
plt.show()

# === EXTRACT CELL FEATURES ===
labeled_mask = label(masks)
props = regionprops_table(labeled_mask, intensity_image=image,
                          properties=['area', 'perimeter', 'mean_intensity'])

cell_sizes = props["area"]
cell_circularities = 4 * np.pi * (props["area"] / (props["perimeter"] ** 2 + 1e-5))
mean_intensities = props["mean_intensity"]

plt.hist(cell_sizes, bins=50, edgecolor="black")
plt.xlabel("Cell Size (pixels)")
plt.ylabel("Count")
plt.title("Distribution of Detected Cell Sizes")
plt.show()

# === DISPLAY CELL STATS ===
print(f"Detected {len(cell_sizes)} cells.")
if len(cell_sizes) > 0:
    print(f"Average cell size: {np.mean(cell_sizes):.2f} pixels.")
    print(f"Average circularity: {np.mean(cell_circularities):.3f}.")
else:
    print("No cells detected, try adjusting the thresholds.")