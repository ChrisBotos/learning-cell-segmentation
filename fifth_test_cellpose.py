import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from skimage import io as skio
from cellpose import models, plot

import pandas as pd
from skimage.measure import regionprops, shannon_entropy, label
from scipy.spatial.distance import pdist, squareform

# === SETTINGS ===
UPSCALE_FACTOR = 1  # Set >1 to upscale (e.g., 2 for 2× zoom), or 1 for no upscaling
CROP_IMAGE = True
enhance_contrast = True
enhance_dim = True
image_path = "/exports/archive/hg-funcgenom-research/IRI_multimodal_project/Stereo-seq_IRI/IRI_regist.tif"

CLAHE_tile_grid_size = (16, 16)

# === SETUP OUTPUT DIRECTORY ===
output_dir = "iri_results_quarter4"
os.makedirs(output_dir, exist_ok=True)

# === GPU CHECK ===
print("=== Checking GPU Availability ===")
torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {torch_device}")
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))


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


def adaptive_gamma_correction(image, min_gamma=1.5, max_gamma=2.5):
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
    image = image[int(12 * h // 16): int(16 * h // 16), int(12 * w // 16): int(16 * w // 16)]
    print(f"Cropped Image Shape: {image.shape}")

# === UPSCALE IMAGE ===
if UPSCALE_FACTOR > 1:
    image = cv2.resize(image, None, fx=UPSCALE_FACTOR, fy=UPSCALE_FACTOR, interpolation=cv2.INTER_LINEAR)
    print(f"Upscaled Image Shape: {image.shape}")

# === SAVE PREPROCESSED IMAGE ===
preprocessed_image_path = os.path.join(output_dir, "preprocessed_image.png")
skio.imsave(preprocessed_image_path, image)
print(f"Saved preprocessed grayscale image: {preprocessed_image_path}")

# === PRINT IMAGE STATS ===
print(f"Image min: {image.min()}, max: {image.max()}, mean: {image.mean()}")

# === CONTRAST ENHANCEMENT (CLAHE) ===
if enhance_contrast:
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=CLAHE_tile_grid_size)
    image = clahe.apply(image)

    contrast_enhanced_image_path = os.path.join(output_dir, "contrast_enhanced_image.png")
    skio.imsave(contrast_enhanced_image_path, image)
    print(f"Saved contrast_enhanced grayscale image: {contrast_enhanced_image_path}")

# === ENHANCE DIM REGIONS ===
if enhance_dim:
    image = adaptive_gamma_correction(image, min_gamma=1.2, max_gamma=1.5)

    gamma_corrected_image_path = os.path.join(output_dir, "gamma_corrected_image.png")
    skio.imsave(gamma_corrected_image_path, image)
    print(f"Saved gamma_corrected grayscale image: {gamma_corrected_image_path}")

# === RUN CELLPOSE SEGMENTATION ===
model = models.Cellpose(model_type="nuclei", gpu=torch.cuda.is_available())

# Invert the image if needed (bright nuclei on dark background).
# image = 255 - image

masks, flows, styles, diams = model.eval(  # TODO maybe add chucks and depth
    image,
    diameter=8 * UPSCALE_FACTOR,  # Approximate average nucleus size in pixels.
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

    cellprob_threshold=-9,  # **Adjusts sensitivity to dim nuclei**.
    # Lower values (e.g., -4) force detection of **low-intensity nuclei**,
    # which is useful for weakly stained or dim cells.
    # Higher values (e.g., 0-2) require **stronger nuclear signals**,
    # potentially missing dim structures.

    resample=True,
    stitch_threshold=0
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
# Load the preprocessed image before CLAHE/gamma for overlay.
original_preprocessed_image = skio.imread(preprocessed_image_path)

mask_overlay = plot.mask_overlay(image, masks, colors=np.random.rand(np.max(masks) + 1, 3))

overlay_path = os.path.join(output_dir, "mask_overlay.png")
skio.imsave(overlay_path, (mask_overlay * 255).astype(np.uint8))
print(f"Saved mask overlay: {overlay_path}")

fig = plt.figure()
plot.show_segmentation(fig=fig, img=image, maski=masks, flowi=flows[0], channels=[0, 0])
fig.savefig(os.path.join(output_dir, "segmentation_debug.png"), dpi=300)

# === DISPLAY FINAL OUTPUT: ALL KEY IMAGES ===
print(f"Detected {np.max(masks)} cells")

fig, axes = plt.subplots(2, 2, figsize=(12, 12))  # 2x2 grid for final output.

# Load images for display
preprocessed_image = skio.imread(preprocessed_image_path)

if enhance_contrast:
    contrast_enhanced_image = skio.imread(contrast_enhanced_image_path)

if enhance_dim:
    gamma_corrected_image = skio.imread(gamma_corrected_image_path)

# Display images
axes[0, 0].imshow(preprocessed_image, cmap="gray")
axes[0, 0].set_title("Preprocessed 8-bit Image")
axes[0, 0].axis("off")

if enhance_contrast:
    axes[0, 1].imshow(contrast_enhanced_image, cmap="gray")
axes[0, 1].set_title("Contrast Enhanced Enhanced Image")
axes[0, 1].axis("off")

if enhance_dim:
    axes[1, 0].imshow(gamma_corrected_image, cmap="gray")
axes[1, 0].set_title("Gamma Corrected Image")
axes[1, 0].axis("off")

axes[1, 1].imshow(mask_overlay)
axes[1, 1].set_title("Cellpose Segmentation Overlay")
axes[1, 1].axis("off")

plt.tight_layout()
final_output_path = os.path.join(output_dir, "final_output_summary.png")
plt.savefig(final_output_path, dpi=300)
plt.show()

print(f"Saved final output summary: {final_output_path}")

# --- Extract Features for Each Nucleus ---

# Use the segmentation mask (masks) and the processed intensity image (image)
regions = regionprops(masks, intensity_image=image)
results = []

for region in regions:
    area = region.area
    perimeter = region.perimeter
    major_axis_length = region.major_axis_length
    minor_axis_length = region.minor_axis_length
    # Circularity: 4*pi*area / (perimeter^2)
    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else np.nan
    eccentricity = region.eccentricity
    solidity = region.solidity
    mean_intensity = region.mean_intensity
    intensity_std = np.std(region.intensity_image)
    # Feret Diameter (if available; otherwise, set as nan)
    try:
        feret_diameter = region.feret_diameter_max
    except AttributeError:
        feret_diameter = np.nan
    # Roughness Index: a crude measure (perimeter divided by square root of area)
    roughness_index = perimeter / np.sqrt(area) if area > 0 else np.nan
    # Texture Entropy: using Shannon entropy on the region's intensity image
    texture_entropy = shannon_entropy(region.intensity_image)

    results.append({
        "Nucleus Area": area,
        "Nucleus Perimeter": perimeter,
        "Nucleus Major Axis Length": major_axis_length,
        "Nucleus Minor Axis Length": minor_axis_length,
        "Nucleus Circularity": circularity,
        "Nucleus Eccentricity": eccentricity,
        "Nucleus Solidity": solidity,
        "Nucleus Intensity Mean": mean_intensity,
        "Nucleus Intensity Standard Deviation": intensity_std,
        "Nucleus Feret Diameter": feret_diameter,
        "Nucleus Roughness Index": roughness_index,
        "Nucleus Texture Entropy": texture_entropy,
    })

df = pd.DataFrame(results)

# --- Compute Nuclei Concentration ---
# (Average nearest-neighbor distance between centroids)
centroids = np.array([region.centroid for region in regions])
if len(centroids) > 1:
    dist_matrix = squareform(pdist(centroids))
    # Exclude self-distances by setting diagonal to infinity
    np.fill_diagonal(dist_matrix, np.inf)
    nearest_neighbor_distances = np.min(dist_matrix, axis=1)
    nuclei_concentration = np.mean(nearest_neighbor_distances)

else:
    nuclei_concentration = np.nan

df["Nuclei Concentration"] = nuclei_concentration

# --- Save the Parameters to a Text File ---
params_txt = os.path.join(output_dir, "segmentation_parameters.txt")
with open(params_txt, "w") as f:
    f.write(df.to_string(index=False))

print(f"Saved segmentation parameters to {params_txt}")

# Calculate mean and median for each parameter
mean_values = df.mean()
median_values = df.median()

# Combine mean and median into a single DataFrame for clarity
summary_df = pd.DataFrame({'Mean': mean_values, 'Median': median_values})

# Define the output directory and file path
summary_file_path = os.path.join(output_dir, "summary_statistics.txt")

# Save the summary statistics to a text file
summary_df.to_csv(summary_file_path, sep='\t', index=True)

print(f"Saved summary statistics to {summary_file_path}")