from __future__ import annotations
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class CustomFashionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # (B, 32, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(2),                              # (B, 32, 14, 14)
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # (B, 64, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(2),                              # (B, 64, 7, 7)
            nn.Flatten(),                                 # (B, 3136)
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    # training
    def train_epoch(self, train_loader: DataLoader, criterion: nn.Module, 
                    optimizer: torch.optim.Optimizer, device: torch.device) -> Tuple[float, float]:
        self.train()
        self.to(device)

        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = self(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy
    
    # FedProx training
    def train_epoch_fedprox(self, train_loader: DataLoader, criterion: nn.Module, 
                            optimizer: torch.optim.Optimizer, device: torch.device, 
                            global_params: List[torch.Tensor], mu: float) -> Tuple[float, float]:
        self.train()
        self.to(device)

        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = self(images)

            # Standard cross-entropy loss
            loss = criterion(logits, labels)

            # Proximal term: (mu/2) * sum ||w_layer - w_global_layer||^2
            prox = 0.0
            for w_local, w_global in zip(self.parameters(), global_params):
                prox += ((w_local - w_global.to(device)) ** 2).sum()
            loss = loss + (mu / 2.0) * prox

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
        
        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    # FedSGD: exactly one mini-batch gradient step
    def train_one_step(self, train_loader: DataLoader, criterion: nn.Module, 
                       optimizer: torch.optim.Optimizer, device: torch.device) -> Tuple[float, float]:
        
        self.train()
        self.to(device)

        # Take only the first batch
        images, labels = next(iter(train_loader))
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = self(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        acc = (preds == labels).sum().item() / labels.size(0)
        return loss.item(), acc

    # evaluation
    @torch.no_grad()
    def test_epoch(self, test_loader: DataLoader, criterion: nn.Module, 
                   device: torch.device) -> Tuple[float, float]:
        self.eval()
        self.to(device)

        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            logits = self(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    # get params
    def get_model_parameters(self) -> List[np.ndarray]:
        return [
            val.cpu().numpy()
            for _, val in self.state_dict().items()
        ]

    # set params
    def set_model_parameters(self, params: List[np.ndarray]) -> None:
        keys = list(self.state_dict().keys())
        new_state_dict = {
            key: torch.tensor(val)
            for key, val in zip(keys, params)
        }
        self.load_state_dict(new_state_dict)