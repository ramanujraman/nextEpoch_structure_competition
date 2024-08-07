import torch

# Check for GPU availability
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Print the device information
print(f"Using device: {device}")