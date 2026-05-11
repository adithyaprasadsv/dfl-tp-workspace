# test_model.py  (temporary)
import torch
import torch.nn as nn
from pytorchexample.model import CustomFashionModel
from pytorchexample.data import load_client_data

model = CustomFashionModel()
device = torch.device("cpu")
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

train_loader, val_loader = load_client_data(cid=0, data_dir="./data_test", batch_size=32)

# One training epoch
train_loss, train_acc = model.train_epoch(train_loader, criterion, optimizer, device)
print(f"Train  — loss: {train_loss:.4f}, acc: {train_acc:.4f}")

# One eval epoch
val_loss, val_acc = model.test_epoch(val_loader, criterion, device)
print(f"Val    — loss: {val_loss:.4f},  acc: {val_acc:.4f}")

# Round-trip parameter serialization
params = model.get_model_parameters()
print(f"Params: {len(params)} arrays, first shape: {params[0].shape}")

model2 = CustomFashionModel()
model2.set_model_parameters(params)
params2 = model2.get_model_parameters()
print("Round-trip OK:", all(
    (a == b).all() for a, b in zip(params, params2)
))