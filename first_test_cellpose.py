import numpy as np
import matplotlib.pyplot as plt
from cellpose import models, io
from skimage import io as skio

image_path = "/DAPI_bIRI2.tif"  # Your uploaded file
image = skio.imread(image_path)

# Convert to grayscale if needed
if image.ndim == 3:
    image = np.mean(image, axis=-1)

plt.imshow(image, cmap='gray')
plt.title("Original DAPI Image")
plt.show()

