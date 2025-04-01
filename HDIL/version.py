import torch

print("PyTorch 版本:", torch.__version__)
print("CUDA 可用性:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA 版本:", torch.version.cuda)
