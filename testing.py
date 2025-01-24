import torch
import cupy as cp

# PyTorch GPU Check
print("Torch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
print("Torch GPU Name:", torch.cuda.get_device_name())

# CuPy GPU Check
print("CuPy Version:", cp.__version__)
print("CuPy GPU Count:", cp.cuda.runtime.getDeviceCount())  # Works
print("CuPy GPU Name:", cp.cuda.runtime.getDeviceProperties(0)["name"])

import torch
print(torch.__version__)                 # Check PyTorch version
print(torch.version.cuda)                # Check CUDA version PyTorch was compiled with
print(torch.cuda.is_available())         # Check if CUDA is available
print(torch.backends.cudnn.enabled)      # Check if cuDNN is available
print(torch.cuda.device_count())         # Number of GPUs detected
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))  # GPU name

