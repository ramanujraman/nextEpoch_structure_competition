from src.data import get_dataloaders
from src.model import RNA_net
from src.util import compute_f1, compute_precision, compute_recall, plot_structures
from src.submission_formatter import format_submission

import torch
import torch.nn as nn

# Check for M1 GPU (MPS device)
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

train_loader, val_loader, test_loader = get_dataloaders(batch_size=8, max_length=70, split=0.8, max_data=1000)

# Init model, loss function, optimizer
model = RNA_net(embedding_dim=64).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([300], device=device))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
train_losses = []
valid_losses = []
f1s_train = []
f1s_valid = []

for epoch in range(num_epochs):
    model.train()
    loss_train = 0.0
    f1_train = 0.0

    for batch in train_loader:
        x = batch["sequence"].to(device)  # (N, L)
        y = batch["structure"].to(device)  # (N, L, L)

        y_pred = model(x)

        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_train += loss.item()
        f1_train += compute_f1(y_pred, y)

    train_losses.append(loss_train / len(train_loader))
    f1s_train.append(f1_train / len(train_loader))

    model.eval()
    loss_valid = 0.0
    f1_valid = 0.0

    with torch.no_grad():
        for batch in val_loader:
            x = batch["sequence"].to(device)  # (N, L)
            y = batch["structure"].to(device)  # (N, L, L)

            y_pred = model(x)

            loss = criterion(y_pred, y)
            loss_valid += loss.item()
            f1_valid += compute_f1(y_pred, y)

    valid_losses.append(loss_valid / len(val_loader))
    f1s_valid.append(f1_valid / len(val_loader))

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Valid Loss: {valid_losses[-1]:.4f}, F1 Train: {f1s_train[-1]:.2f}, F1 Valid: {f1s_valid[-1]:.2f}")

# Test loop
model.eval()
structures = []

with torch.no_grad():
    for batch in test_loader:
        x = batch["sequence"].to(device)  # (N, L)
        y_pred = model(x)

        for seq, pred in zip(batch["sequence"], y_pred):
            structure = (pred > 0.5).type(torch.int).cpu()  # Convert to binary and move to CPU
            structures.append(structure)

format_submission(test_loader.dataset.sequences, test_loader.dataset.structures, structures, 'test_pred.csv')
