import os
import numpy as np
import torch
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io as skio
from skimage.measure import regionprops_table, label
from cellpose import models, plot

# === SETTINGS ===
UPSCALE_FACTOR = 1  # Set >1 to upscale (e.g., 2 for 2× zoom), or 1 for no upscaling
NUM_PARTS = 8  # Number of regions to split the image into
EXPERIMENT_NAME = "experiment_02"  # User-defined experiment name
image_path = "DAPI_bIRI2.tif"

# === SETUP OUTPUT DIRECTORY ===
output_dir = os.path.join("results", EXPERIMENT_NAME)
os.makedirs(output_dir, exist_ok=True)

# === GPU CHECK ===
print("=== Checking GPU Availability ===")
torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {torch_device}")
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))

# === LOAD IMAGE ===
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

# === UPSCALE IMAGE ===
if UPSCALE_FACTOR > 1:
    image = cv2.resize(image, None, fx=UPSCALE_FACTOR, fy=UPSCALE_FACTOR, interpolation=cv2.INTER_LINEAR)
    print(f"Upscaled Image Shape: {image.shape}")

# TODO remove
h, w = image.shape
image = image[h // 2:,w // 2 : ]
print(f"Cropped Image Shape: {image.shape}")

# === SPLIT IMAGE INTO REGIONS ===
h, w = image.shape
num_rows = int(np.sqrt(NUM_PARTS))
num_cols = int(NUM_PARTS / num_rows)
patch_h = h // num_rows
patch_w = w // num_cols

masks_combined = np.zeros_like(image, dtype=np.uint32)  # To store all segmented masks

# === CELLPOSE SEGMENTATION ===
model = models.Cellpose(model_type='nuclei', gpu=torch.cuda.is_available())

for i in range(num_rows):
    for j in range(num_cols):
        y1, y2 = i * patch_h, (i + 1) * patch_h
        x1, x2 = j * patch_w, (j + 1) * patch_w
        patch = image[y1:y2, x1:x2]

        print(f"Processing patch ({i}, {j}) of shape {patch.shape}")

        # === CONTRAST ENHANCEMENT (CLAHE) ===
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(16, 16))
        patch = clahe.apply(patch)

        # === CELLPOSE SEGMENTATION ===
        masks, _, _, _ = model.eval(
            patch,
            diameter=5 * UPSCALE_FACTOR,
            channels=[0, 0],
            flow_threshold=0.8,
            cellprob_threshold=-1,
            resample=True
        )

        # Adjust mask indices to ensure uniqueness
        masks += masks_combined.max()

        # Store the segmented region in the final mask
        masks_combined[y1:y2, x1:x2] = masks

        # Save patch mask
        patch_mask_path = os.path.join(output_dir, f"segmentation_mask_{i}_{j}.png")
        skio.imsave(patch_mask_path, masks.astype(np.uint16))

# === SAVE FINAL COMBINED MASK ===
print("Combining the masks of all parts...")
final_mask_path = os.path.join(output_dir, "final_segmentation_mask.png")
skio.imsave(final_mask_path, masks_combined.astype(np.uint16))


# === RELABEL MASKS ===
labeled_mask = label(masks_combined)

# === OPTIMIZE COLOR ASSIGNMENT ===
unique_labels = np.unique(labeled_mask)
color_map = np.random.rand(len(unique_labels), 3)  # Assign colors only to existing labels

# === CREATE & SAVE MASK OVERLAY ===
print("Creating and saving mask overlay...")
mask_overlay = plot.mask_overlay(image, labeled_mask, colors=color_map)
overlay_path = os.path.join(output_dir, "final_mask_overlay.png")
skio.imsave(overlay_path, (mask_overlay * 255).astype(np.uint8))
print(f"Saved final mask overlay: {overlay_path}")



# === EXTRACT CELL FEATURES ===
print("Extracting cell features...")
props = regionprops_table(labeled_mask, intensity_image=image,
                          properties=['area', 'perimeter', 'mean_intensity'])

cell_sizes = props["area"]

# === FIXED CIRCULARITY CALCULATION ===
cell_circularities = np.zeros_like(cell_sizes, dtype=np.float32)
valid_indices = props["perimeter"] > 0  # Avoid division by zero
cell_circularities[valid_indices] = (
    4 * np.pi * props["area"][valid_indices] / (props["perimeter"][valid_indices] ** 2)
)

mean_intensities = props["mean_intensity"]

# === SAVE CELL DATA TO CSV ===
csv_path = os.path.join(output_dir, "cell_features.csv")
df = pd.DataFrame({"cell_size": cell_sizes, "circularity": cell_circularities, "mean_intensity": mean_intensities})
df.to_csv(csv_path, index=False)
print(f"Saved cell feature data to: {csv_path}")

# === HISTOGRAMS: CELL SIZE & CIRCULARITY ===
plt.figure(figsize=(12, 5))

# Cell Size Distribution
plt.subplot(1, 2, 1)
plt.hist(cell_sizes, bins=50, edgecolor="black")
plt.xlabel("Cell Size (pixels)")
plt.ylabel("Count")
plt.title("Distribution of Detected Cell Sizes")

# Circularity Distribution
plt.subplot(1, 2, 2)
plt.hist(cell_circularities, bins=50, edgecolor="black")
plt.xlabel("Circularity")
plt.ylabel("Count")
plt.title("Distribution of Cell Circularities")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "cell_stats_histograms.png"), dpi=300)
plt.show()

# === DISPLAY CELL STATS ===
print(f"Detected {len(cell_sizes)} cells.")
if len(cell_sizes) > 0:
    print(f"Average cell size: {np.mean(cell_sizes):.2f} pixels.")
    print(f"Average circularity: {np.mean(cell_circularities):.3f}.")
else:
    print("No cells detected, try adjusting the thresholds.")
