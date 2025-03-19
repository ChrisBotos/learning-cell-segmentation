import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from skimage import io as skio

# Import SPArrOW’s modules.
# (Ensure you have installed sparrow via pip from GitHub and its dependencies.)
from sparrow import segmentation, visualization

# === SETTINGS ===
UPSCALE_FACTOR = 1       # Set >1 to upscale (e.g., 2 for 2× zoom); 1 for no upscaling.
CROP_IMAGE = True        # Whether to crop the image.
ENHANCE_DIM = True       # Whether to apply adaptive gamma correction.
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

# === UTILITY FUNCTIONS ===

def convert_16bit_to_8bit(image):
    """
    Converts a 16-bit grayscale image to 8-bit using contrast stretching.
    Uses the 1st and 99th percentile to reduce outlier influence.
    """
    if image.dtype != np.uint16:
        return image  # Return unchanged if not 16-bit.
    p1, p99 = np.percentile(image, (1, 99))
    if p99 - p1 == 0:
        p1, p99 = image.min(), image.max()
    image_8bit = np.clip((image - p1) / (p99 - p1) * 255, 0, 255).astype(np.uint8)
    print("Converted 16-bit image to 8-bit with contrast stretching.")
    return image_8bit

def adaptive_gamma_correction(image, min_gamma=1.2, max_gamma=2.5):
    """
    Applies adaptive gamma correction based on median brightness.
    Darker images get a higher gamma; brighter images get lower gamma.
    """
    median_intensity = np.median(image)
    norm_intensity = median_intensity / 255.0
    gamma = max_gamma - (max_gamma - min_gamma) * norm_intensity
    gamma = np.clip(gamma, min_gamma, max_gamma)
    print(f"Applying Gamma Correction with γ = {gamma:.2f}")
    lookup_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype("uint8")
    corrected_image = cv2.LUT(image, lookup_table)
    return corrected_image

# === LOAD AND PREPROCESS IMAGE ===
image = skio.imread(image_path)
print(f"Original Image: dtype={image.dtype}, shape={image.shape}")

# Handle RGBA images: remove alpha channel if present.
if image.ndim == 3 and image.shape[-1] == 4:
    image = image[:, :, :3]
    print("Removed alpha channel.")

# Convert 16-bit to 8-bit if needed.
if image.dtype == np.uint16:
    image = convert_16bit_to_8bit(image)
    cv2.imwrite("cell_image_8bit.tif", image)
    print("Saved converted 8-bit image.")

# Convert to grayscale if the image is RGB.
if image.ndim == 3:
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    print("Converted image to grayscale.")

# Crop the image if enabled.
if CROP_IMAGE:
    h, w = image.shape
    # Example cropping: central vertical band (adjust as needed).
    image = image[5*h//8: 6*h//8, 4*w//8 : 5*w//8]
    print(f"Cropped Image Shape: {image.shape}")

# Upscale the image if required.
if UPSCALE_FACTOR > 1:
    image = cv2.resize(image, None, fx=UPSCALE_FACTOR, fy=UPSCALE_FACTOR, interpolation=cv2.INTER_LINEAR)
    print(f"Upscaled Image Shape: {image.shape}")

# Save the preprocessed image.
preprocessed_image_path = os.path.join(output_dir, "preprocessed_image.png")
skio.imsave(preprocessed_image_path, image)
print(f"Saved preprocessed grayscale image: {preprocessed_image_path}")

print(f"Image stats - min: {image.min()}, max: {image.max()}, mean: {image.mean()}")

# === CONTRAST ENHANCEMENT (CLAHE) ===
clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
image = clahe.apply(image)
contrast_enhanced_image_path = os.path.join(output_dir, "contrast_enhanced_image.png")
skio.imsave(contrast_enhanced_image_path, image)
print(f"Saved contrast enhanced image: {contrast_enhanced_image_path}")

# === ENHANCE DIM REGIONS (Adaptive Gamma Correction) ===
if ENHANCE_DIM:
    image = adaptive_gamma_correction(image, min_gamma=1.2, max_gamma=1.5)
    gamma_corrected_image_path = os.path.join(output_dir, "gamma_corrected_image.png")
    skio.imsave(gamma_corrected_image_path, image)
    print(f"Saved gamma corrected image: {gamma_corrected_image_path}")

# === RUN SPArrOW SEGMENTATION ===
# Pass the preprocessed image to SPArrOW’s segmentation function.
# Additional parameters can be added based on SPArrOW’s API.
print("Running SPArrOW segmentation...")
segmentation_result = segmentation.segment_image(image, use_gpu=torch.cuda.is_available())
masks = segmentation_result.mask  # Assuming the result has a 'mask' attribute.

# === SAVE SEGMENTATION MASK ===
mask_path = os.path.join(output_dir, "segmentation_mask.png")
skio.imsave(mask_path, masks.astype(np.uint16))
print(f"Saved segmentation mask: {mask_path}")

# === NORMALIZE & SAVE MASK FOR VISUALIZATION ===
normalized_mask = (masks / masks.max() * 255).astype(np.uint8) if masks.max() > 0 else masks
cv2.imwrite(os.path.join(output_dir, "segmentation_mask_visual.png"), normalized_mask)
print("Saved normalized mask for visualization.")

# === CREATE & SAVE MASK OVERLAY ===
# Using SPArrOW’s visualization (or fallback to a similar overlay function)
mask_overlay = visualization.plot_segmentation(image, masks, random_colors=True)
overlay_path = os.path.join(output_dir, "mask_overlay.png")
skio.imsave(overlay_path, (mask_overlay * 255).astype(np.uint8))
print(f"Saved mask overlay: {overlay_path}")

# === DISPLAY FINAL OUTPUT ===
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# Reload images for display.
preprocessed_image_disp = skio.imread(preprocessed_image_path)
contrast_enhanced_image_disp = skio.imread(contrast_enhanced_image_path)
if ENHANCE_DIM:
    gamma_corrected_image_disp = skio.imread(gamma_corrected_image_path)

axes[0, 0].imshow(preprocessed_image_disp, cmap="gray")
axes[0, 0].set_title("Preprocessed Image")
axes[0, 0].axis("off")

axes[0, 1].imshow(contrast_enhanced_image_disp, cmap="gray")
axes[0, 1].set_title("Contrast Enhanced")
axes[0, 1].axis("off")

if ENHANCE_DIM:
    axes[1, 0].imshow(gamma_corrected_image_disp, cmap="gray")
    axes[1, 0].set_title("Gamma Corrected")
    axes[1, 0].axis("off")

axes[1, 1].imshow(mask_overlay)
axes[1, 1].set_title("Segmentation Overlay")
axes[1, 1].axis("off")

plt.tight_layout()
final_output_path = os.path.join(output_dir, "final_output_summary.png")
plt.savefig(final_output_path, dpi=300)
plt.show()
print(f"Saved final output summary: {final_output_path}")
