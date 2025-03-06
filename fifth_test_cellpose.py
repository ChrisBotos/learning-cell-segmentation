import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from skimage import io as skio
from skimage.measure import regionprops_table, label
from skimage.morphology import white_tophat, disk
from skimage.restoration import denoise_nl_means, estimate_sigma
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import watershed
from cellpose import models, plot

"""TITLE: DAPI-Stained Nuclei Segmentation Pipeline"""

# === SETTINGS ===
UPSCALE_FACTOR = 1  # Set >1 to upscale (e.g., 2 for 2× zoom), or 1 for no upscaling.
CROP_IMAGE = True  # Enable cropping to focus on a specific region.
enhance_dim = True  # Whether to enhance dim regions using gamma correction.
image_path = "DAPI_bIRI2.tif"  # Path to the input image.

# === SETUP OUTPUT DIRECTORY ===
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# === CHECK GPU AVAILABILITY ===
torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {torch_device}")
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))

# === LOAD IMAGE ===
image = skio.imread(image_path)
print(f"Original Image: dtype={image.dtype}, shape={image.shape}")

# === CROP IMAGE (IF ENABLED) ===
if CROP_IMAGE:
    h, w = image.shape[:2]
    image = image[5*h//8: 6*h//8, 4*w//8: 5*w//8]  # Cropping to focus on region of interest.
    print(f"Cropped Image Shape: {image.shape}")

# === HANDLE IMAGE FORMAT ===
if image.ndim == 3 and image.shape[-1] == 4:
    image = image[:, :, :3]  # Remove alpha channel if present.
    print("Removed alpha channel.")

if image.dtype == np.uint16:
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    print("Converted 16-bit image to 8-bit.")

if image.ndim == 3:
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    print("Converted image to grayscale.")

# Save preprocessed image
skio.imsave(os.path.join(output_dir, "01_preprocessed_image.png"), image)

# === UPSCALE IMAGE (IF ENABLED) ===
if UPSCALE_FACTOR > 1:
    image = cv2.resize(image, None, fx=UPSCALE_FACTOR, fy=UPSCALE_FACTOR, interpolation=cv2.INTER_LINEAR)
    skio.imsave(os.path.join(output_dir, "02_upscaled_image.png"), image)

# # === DENOISING (Preserve fine details) ===
# sigma_est = np.mean(estimate_sigma(image))  # Estimate noise level.
# image = denoise_nl_means(image, h=1.15 * sigma_est, fast_mode=True, patch_size=5, patch_distance=6, preserve_range=True)
# image = (image * 255).astype(np.uint8)  # Convert back to 8-bit format.
#
# # Save denoised image
# skio.imsave(os.path.join(output_dir, "03_denoised_image.png"), image)
#
# # === ADAPTIVE BACKGROUND REMOVAL ===
# background = cv2.medianBlur(image, 15)  # Smaller kernel prevents excessive subtraction.
# image = cv2.subtract(image, background)  # Subtract background adaptively.
# image = cv2.addWeighted(image, 1.2, background, -0.2, 0)  # Adjust intensity to retain dim structures.
#
# # Save background-removed image
# skio.imsave(os.path.join(output_dir, "04_background_removed.png"), image)

# === CONTRAST ENHANCEMENT (CLAHE) ===
print("Applying CLAHE for contrast correction.")
clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8, 8))  # Moderate contrast adjustment.
image = clahe.apply(image)

# Save contrast-enhanced image
skio.imsave(os.path.join(output_dir, "05_contrast_enhanced.png"), image)



# === COMPUTE DISTANCE TRANSFORM (For Watershed) ===

# Step 1: Create a more robust binary mask using adaptive thresholding.
binary_mask = cv2.adaptiveThreshold(
    image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5
)

# Step 2: Compute distance transform on the binary mask.
dist_transform = distance_transform_edt(binary_mask)

# Step 3: Normalize distance transform to maintain nuclei structure.
dist_transform = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX)

# Step 4: Apply Gaussian blur to prevent oversized regions.
dist_transform = cv2.GaussianBlur(dist_transform.astype(np.uint8), (3, 3), 0)

# Save fixed distance transform image.
skio.imsave(os.path.join(output_dir, "06_distance_transform.png"), dist_transform)

# === GENERATE FOREGROUND & BACKGROUND MASKS ===

# Create a refined foreground mask by thresholding the distance map.
_, foreground = cv2.threshold(dist_transform, 30, 255, cv2.THRESH_BINARY)

# Reduce background expansion (prevent oversized regions).
kernel = np.ones((2, 2), np.uint8)  # Smaller kernel prevents over-expansion.
background = cv2.dilate(foreground, kernel, iterations=1)

# Erode to separate overlapping nuclei before watershed.
background = cv2.erode(background, kernel, iterations=1)

# Save refined background mask.
skio.imsave(os.path.join(output_dir, "07_background_mask.png"), background)

# === WATERSHED SEGMENTATION ===

# Compute unknown regions and markers for watershed.
unknown = cv2.subtract(background, foreground)
_, markers = cv2.connectedComponents(foreground)

# Increment marker values for separation.
markers = markers + 1
markers[unknown == 255] = 0

# Apply watershed with refined distance transform.
watershed_labels = watershed(-dist_transform, markers, mask=image)

# Save watershed labels
skio.imsave(os.path.join(output_dir, "08_watershed_labels.png"), watershed_labels.astype(np.uint8))

# === SEGMENTATION ===
model = models.Cellpose(model_type='nuclei', gpu=torch.cuda.is_available())
masks, _, _, _ = model.eval(image, diameter=5 * UPSCALE_FACTOR, channels=[0, 0], flow_threshold=0.9, cellprob_threshold=-4, resample=True)

# Save segmentation mask
skio.imsave(os.path.join(output_dir, "09_segmentation_mask.png"), masks.astype(np.uint16))

# === FEATURE EXTRACTION ===
labeled_mask = label(masks)
props = regionprops_table(labeled_mask, intensity_image=image, properties=['area', 'perimeter', 'mean_intensity'])
cell_sizes = props["area"]
valid_indices = props["perimeter"] > 5
cell_circularities = np.zeros_like(cell_sizes, dtype=np.float32)
cell_circularities[valid_indices] = (4 * np.pi * props["area"][valid_indices] / (props["perimeter"][valid_indices] ** 2))
cell_circularities[(cell_circularities < 0) | (cell_circularities > 1)] = np.nan
mean_intensities = props["mean_intensity"]

# === POST-PROCESSING ===
valid_cells = (cell_sizes > 10) & (cell_sizes < 500)
valid_labels = np.where(valid_cells)[0] + 1
filtered_masks = np.isin(masks, valid_labels).astype(np.uint8) * masks

# Apply morphological operations to refine segmentation
filtered_masks = cv2.morphologyEx(filtered_masks.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
filtered_masks = cv2.dilate(filtered_masks, kernel, iterations=1)

# Save final processed mask
skio.imsave(os.path.join(output_dir, "10_postprocessed_mask.png"), filtered_masks.astype(np.uint16))

# === DISPLAY CELL STATS ===
print(f"Detected {len(cell_sizes)} cells.")
if len(cell_sizes) > 0:
    print(f"Average cell size: {np.mean(cell_sizes):.2f} pixels.")
    print(f"Average circularity: {np.nanmean(cell_circularities):.2f}")
    print(f"Average intensity: {np.mean(mean_intensities):.2f}")
else:
    print("No cells detected, try adjusting segmentation thresholds.")
