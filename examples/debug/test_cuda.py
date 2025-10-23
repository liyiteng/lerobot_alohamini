import torch
print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("torch.cuda.is_available",torch.cuda.is_available())  # Should return True
print("torch.cuda.device_coun",torch.cuda.device_count())  # Should return the number of GPUs
print("torch.cuda.get_device_name(0)",torch.cuda.get_device_name(0))  # Display the GPU name