import numpy as np
import cupy as cp  # CuPy for GPU-accelerated arrays
import matplotlib.pyplot as plt
import torch
from cellpose import models, io, plot
from skimage import io as skio
import cv2  # For additional preprocessing
from skimage.measure import regionprops, label
print("Imported")

print(torch.cuda.is_available())  # Should print True
print(torch.cuda.device_count())  # Should print 1 or more
print(torch.cuda.get_device_name(0))  # Should print your GPU name
print(torch.cuda.current_device())  # Should print 0 if using first GPU

# === Configuration ===
image_path = "DAPI_bIRI2.tif"  # Your image file
crop_image = False  # Set to 'False' to use the full image
crop_size = (1000, 1000)  # (height, width)


# === Load Image ===
print("Loading Image...")
image = skio.imread(image_path)

# Debugging: Check image depth
print(f"Original image dtype: {image.dtype}, min/max: {image.min()}, {image.max()}")

# === Convert to 8-bit if necessary ===
if image.dtype == np.uint16:  # Check if it's 16-bit
    image = (image / 256).astype(np.uint8)  # Convert to 8-bit (0-255)
    print("Converted 16-bit image to 8-bit.")

# Convert to grayscale if it's RGB
if image.ndim == 3:
    image = np.mean(image, axis=-1).astype(np.uint8)

# === OPTIONAL CROPPING ===
if crop_image:
    h, w = image.shape[:2]  # Ensure we only take height and width
    crop_h, crop_w = crop_size  # Unpack the crop size tuple

    # Ensure the crop size does not exceed the image size
    crop_h = min(crop_h, h)
    crop_w = min(crop_w, w)

    # Compute start indices for center cropping
    start_h = (h - crop_h) // 2
    start_w = (w - crop_w) // 2

    # Crop the center of the image
    image = image[start_h:start_h + crop_h, start_w:start_w + crop_w]

# Debugging: Check final image
print(f"Final image shape: {image.shape}, dtype: {image.dtype}, min/max: {image.min()}, {image.max()}")

# === Contrast Enhancement ===
image = cv2.equalizeHist(image)

# Save the cropped & processed image
output_path = "processed_image.png"
cv2.imwrite(output_path, image)
print(f"Cropped image saved to {output_path}")

# === Display Processed Image ===
plt.figure(figsize=(6, 6))
plt.imshow(image, cmap='gray')  # Show cropped image
plt.title(f"Processed Image {image.shape}")
plt.axis("off")
plt.show()

# === Check GPU Availability ===
torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {torch_device}")
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))

# Convert to CuPy array if CUDA is available
if torch.cuda.is_available():
    image_cp = cp.array(image, dtype=cp.uint8)  # Move image to GPU
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    image_cp = image  # Use NumPy if no GPU is found
    print("No GPU available, using CPU.")

# === Running the Cell Segmentation ===
model = models.Cellpose(model_type='nuclei', gpu=torch.cuda.is_available())  # Ensure GPU usage

masks, flows, styles, diams = model.eval(
    cp.asnumpy(image_cp),  # Convert back to NumPy for Cellpose
    diameter=None,  # Auto-detects nucleus size
    channels=[0, 0],  # Grayscale image
    flow_threshold=0.005,  # Adjust segmentation sensitivity
    cellprob_threshold=-7  # Adjusts minimum probability for detection
)

# === Visualization of Results ===
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Show original image
axes[0].imshow(cp.asnumpy(image_cp), cmap='gray')
axes[0].set_title("Processed DAPI Image")
axes[0].axis("off")

# Show segmented masks with outlines
mask_overlay = plot.mask_overlay(cp.asnumpy(image_cp), masks)
axes[1].imshow(mask_overlay)
axes[1].set_title("Cellpose Segmentation Mask")
axes[1].axis("off")

plt.show()

# === Save the Segmentation Mask ===
mask_path = image_path.replace(".tif", "_seg.tif")
io.masks_flows_to_seg(cp.asnumpy(image_cp), masks, flows, mask_path)
print(f"Segmentation mask saved to: {mask_path}")

# === Extract Cell Features ===
labeled_mask = label(masks)  # Label each nucleus
props = regionprops(labeled_mask, intensity_image=cp.asnumpy(image_cp))  # Convert back to NumPy

# Extract properties
cell_sizes = [prop.area for prop in props]  # Size in pixels
cell_circularities = [4 * np.pi * (prop.area / (prop.perimeter**2 + 1e-5)) for prop in props]  # Circularity
mean_intensities = [prop.mean_intensity for prop in props]  # Mean intensity per cell

# Display statistics
print(f"Detected {len(props)} cells")
if len(cell_sizes) > 0:
    print(f"Average cell size: {np.mean(cell_sizes):.2f} pixels")
    print(f"Average circularity: {np.mean(cell_circularities):.3f}")
else:
    print("No cells detected, try adjusting the thresholds.")
