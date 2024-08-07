from src.data import get_dataloaders
from src.model import RNA_net
from src.util import compute_f1, compute_precision, compute_recall, plot_structures
from src.submission_formatter import format_submission

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

# Check for MPS device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Data loaders
train_loader, val_loader, test_loader = get_dataloaders(batch_size=8, max_length=70, split=0.8, max_data=1000)

# Model, loss function, optimizer, and scheduler
model = RNA_net(embedding_dim=64).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([300.0]).to(device))
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

# Training loop
num_epochs = 10
train_losses = []
valid_losses = []
f1s_train = []
f1s_valid = []

best_valid_f1 = 0.0

for epoch in range(num_epochs):
    model.train()
    loss_train = 0.0
    f1_train = 0.0

    # Training phase
    for batch in train_loader:
        x = batch["sequence"].to(device)  # (N, L)
        y = batch['structure'].to(device)  # (N, L, L)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        loss_train += loss.item()
        f1_train += compute_f1(y_pred, y)

    train_losses.append(loss_train / len(train_loader))
    f1s_train.append(f1_train / len(train_loader))

    # Validation phase
    model.eval()
    loss_valid = 0.0
    f1_valid = 0.0

    with torch.no_grad():
        for batch in val_loader:
            x = batch["sequence"].to(device)  # (N, L)
            y = batch['structure'].to(device)  # (N, L, L)

            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss_valid += loss.item()
            f1_valid += compute_f1(y_pred, y)

    valid_losses.append(loss_valid / len(val_loader))
    f1s_valid.append(f1_valid / len(val_loader))

    scheduler.step(loss_valid / len(val_loader))

    if f1_valid / len(val_loader) > best_valid_f1:
        best_valid_f1 = f1_valid / len(val_loader)
        torch.save(model.state_dict(), 'best_model.pth')

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Train F1: {f1s_train[-1]:.2f}, "
          f"Valid Loss: {valid_losses[-1]:.4f}, Valid F1: {f1s_valid[-1]:.2f}")

# Load the best model for test phase
model.load_state_dict(torch.load('best_model.pth'))

# Test loop
structures = []
for sequence in test_loader[1]:  # Assuming test_loader[1] contains sequences
    sequence = sequence.to(device)
    with torch.no_grad():
        structure = (model(sequence.unsqueeze(0)).squeeze(0) > 0.5).type(torch.int).cpu()  # Has to be shape (L, L)
    structures.append(structure)

format_submission(test_loader[0], test_loader[1], structures, 'test_pred.csv')  # Ensure test_loader[0] contains indices
