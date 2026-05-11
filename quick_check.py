import torch
import torchvision
import flwr as fl
import numpy
import prettytable
import tomli
print("Flower:", fl.__version__)
print("Torch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("CUDA␣available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))