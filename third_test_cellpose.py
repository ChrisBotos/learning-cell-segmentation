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
enhance_dim = True
image_path = "DAPI_bIRI2.tif"

# === SETUP OUTPUT DIRECTORY ===
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# === GPU CHECK ===
print("=== Checking GPU Availability ===")
torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {torch_device}")
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))

import numpy as np
import cv2


def convert_16bit_to_8bit(image):
    """
    Converts a 16-bit grayscale image to 8-bit using contrast stretching.

    - Uses the 1st and 99th percentile instead of min/max to reduce the effect of outliers.
    - Ensures that images with zero intensity variation do not cause division errors.

    Parameters:
        image (numpy.ndarray): 16-bit grayscale image.

    Returns:
        numpy.ndarray: 8-bit grayscale image.
    """

    if image.dtype != np.uint16:
        return image  # Return unchanged if already 8-bit or another format.

    # Compute percentiles for contrast stretching (ignore extreme outliers)
    p1, p99 = np.percentile(image, (1, 99))

    # Prevent division by zero if the image has uniform intensity
    if p99 - p1 == 0:
        p1, p99 = image.min(), image.max()

    # Apply normalization to 8-bit range (0-255)
    image_8bit = np.clip((image - p1) / (p99 - p1) * 255, 0, 255).astype(np.uint8)

    print("Converted 16-bit image to 8-bit with contrast stretching.")

    return image_8bit


import numpy as np
import cv2

def adaptive_gamma_correction(image, min_gamma=1.2, max_gamma=2.5):
    """
    Applies adaptive gamma correction based on image brightness.

    - Computes median pixel intensity to adjust gamma dynamically.
    - Uses a lookup table for efficient transformation.

    Parameters:
        image (numpy.ndarray): 8-bit grayscale image.
        min_gamma (float): Minimum gamma value for bright images.
        max_gamma (float): Maximum gamma value for dark images.

    Returns:
        numpy.ndarray: Gamma-corrected image.
    """

    # Compute median intensity (brightness reference)
    median_intensity = np.median(image)

    # Normalize intensity range (0-255 → 0-1)
    norm_intensity = median_intensity / 255.0

    # Adjust gamma adaptively: Darker images get higher gamma, brighter ones get lower gamma
    gamma = max_gamma - (max_gamma - min_gamma) * norm_intensity

    # Avoid extreme values (ensure gamma is within range)
    gamma = np.clip(gamma, min_gamma, max_gamma)

    print(f"Applying Gamma Correction with γ = {gamma:.2f}")

    # Compute lookup table for fast transformation
    lookup_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype("uint8")

    # Apply gamma correction using the lookup table
    corrected_image = cv2.LUT(image, lookup_table)

    return corrected_image



# === LOAD IMAGE ===
image = skio.imread(image_path)
print(f"Original Image: dtype={image.dtype}, shape={image.shape}")

# === HANDLE RGBA IMAGES ===
if image.ndim == 3 and image.shape[-1] == 4:
    image = image[:, :, :3]  # Remove alpha channel
    print("Removed alpha channel.")

# === CONVERT 16-BIT TO 8-BIT ===
if image.dtype == np.uint16:
    image = convert_16bit_to_8bit(image)
    cv2.imwrite("cell_image_8bit.tif", image)
    print("Converted 16-bit image to 8-bit.")

# === CONVERT TO GRAYSCALE ===
if image.ndim == 3:
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    print("Converted image to grayscale.")

# === CROP IMAGE ===
if CROP_IMAGE:
    h, w = image.shape
    image = image[5*h//8: 6*h//8, 4*w//8 : 5*w//8]
    print(f"Cropped Image Shape: {image.shape}")


# === UPSCALE IMAGE ===
if UPSCALE_FACTOR > 1:
    image = cv2.resize(image, None, fx=UPSCALE_FACTOR, fy=UPSCALE_FACTOR, interpolation=cv2.INTER_LINEAR)
    print(f"Upscaled Image Shape: {image.shape}")

# === SAVE PREPROCESSED IMAGE ===
gray_image_path = os.path.join(output_dir, "preprocessed_image.png")
skio.imsave(gray_image_path, image)
print(f"Saved preprocessed grayscale image: {gray_image_path}")

# === PRINT IMAGE STATS ===
print(f"Image min: {image.min()}, max: {image.max()}, mean: {image.mean()}")

# === CONTRAST ENHANCEMENT (CLAHE) ===
clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
image = clahe.apply(image)

# === ENHANCE DIM REGIONS ===
if enhance_dim:
    image = adaptive_gamma_correction(image, min_gamma = 1.2, max_gamma=2.5)


# === RUN CELLPOSE SEGMENTATION ===
model = models.Cellpose(model_type="nuclei",gpu=torch.cuda.is_available())

masks, flows, styles, diams = model.eval(
    image,
    diameter=5 * UPSCALE_FACTOR,  # Approximate average nucleus size in pixels.
    # Larger values make Cellpose detect bigger objects,
    # while smaller values detect finer details.

    channels=[0, 0],  # Specifies that the input is a **single-channel grayscale image**.
    # First value: The channel containing nuclei (0 = grayscale).
    # Second value: Optional cytoplasm channel (0 = none, used for multi-channel images).

    flow_threshold=0.9,  # **Controls boundary sharpness**.
    # Higher values (closer to 1) enforce well-defined edges,
    # making the model stricter in separating touching cells.
    # Lower values (e.g., 0.4-0.6) allow more flexible boundaries
    # but might cause over-segmentation.

    cellprob_threshold=-4,  # **Adjusts sensitivity to dim nuclei**.
    # Lower values (e.g., -4) force detection of **low-intensity nuclei**,
    # which is useful for weakly stained or dim cells.
    # Higher values (e.g., 0-2) require **stronger nuclear signals**,
    # potentially missing dim structures.
)

# === SAVE MASK IMAGE ===
mask_path = os.path.join(output_dir, "segmentation_mask.png")
skio.imsave(mask_path, masks.astype(np.uint16))  # Save as 16-bit mask

# === NORMALIZE & SAVE MASK FOR VISUALIZATION ===
# The original `masks` array contains integer labels for segmented objects (cells),
# where each unique value represents a different cell.
# However, these values are **not** in the 0-255 grayscale range, making it
# difficult to visualize directly.

# Step 1: Normalize the mask to 0-255 for visualization.
# - `masks.max()` gives the highest label value in the mask.
# - Dividing by `masks.max()` scales all labels to the 0-1 range.
# - Multiplying by 255 brings it to 8-bit grayscale format (0-255).
# - `astype(np.uint8)` ensures it is saved correctly as an 8-bit image.
normalized_mask = (masks / masks.max() * 255).astype(np.uint8) if masks.max() > 0 else masks

# Step 2: Save the normalized mask image.
# This allows visualization in standard image viewers, which expect 8-bit images.
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
valid_indices = props["perimeter"] > 5  # Ignore objects with tiny perimeters
cell_circularities = np.zeros_like(cell_sizes, dtype=np.float32)
cell_circularities[valid_indices] = (
    4 * np.pi * props["area"][valid_indices] / (props["perimeter"][valid_indices] ** 2)
)

# Set unrealistic circularities to NaN
cell_circularities[(cell_circularities < 0) | (cell_circularities > 1)] = np.nan

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
    print(f"Average circularity: {np.mean(cell_circularities[(cell_circularities >= 0) & (cell_circularities <= 1)])}")
else:
    print("No cells detected, try adjusting the thresholds.")