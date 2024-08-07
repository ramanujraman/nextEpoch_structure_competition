from src.data import get_dataloaders
from src.model import RNA_net
from src.util import compute_f1, compute_precision, compute_recall, plot_structures
from src.submission_formatter import format_submission

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
train_loader, val_loader, test_loader = get_dataloaders(batch_size=2, max_length=50, split=0.8, max_data=1000)

# Init model, loss function, optimizer
model = RNA_net(embedding_dim=32).to(device)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):

    model.train()
    loss_epoch = 0.0
    metric_epoch = 0.0

    for batch in train_loader:

        x = batch["sequence"].to(device) # (N, L)
        y = batch['structure'].to(device) # (N, L, L)

        y_pred = model(x)

        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()
        metric_epoch += compute_f1(y_pred, y)

    print(f"Epoch {epoch+1}: Loss {loss_epoch/len(train_loader):.4f}, F1 score {metric_epoch/len(train_loader):.4f}")

    # Validation loop
    model.eval()
    val_loss = 0.0
    val_metric = 0.0
    with torch.no_grad():
        for batch in val_loader:
            x = batch["sequence"].to(device) # (N, L)
            y = batch['structure'].to(device) # (N, L, L)

            y_pred = model(x)

            loss = criterion(y_pred, y)
            val_loss += loss.item()
            val_metric += compute_f1(y_pred, y)

    print(f"Validation Loss: {val_loss/len(val_loader):.4f}, Validation F1 score: {val_metric/len(val_loader):.4f}")

# Test loop
model.eval()
structures = []
with torch.no_grad():
    for sequence in test_loader[1]:
        x = sequence.to(device) # Move sequence to device
        y_pred = model(x)
        structure = (torch.sigmoid(y_pred) > 0.5).type(torch.int) # Convert logits to binary predictions
        structures.append(structure.cpu()) # Move structure back to CPU

format_submission(test_loader[0], test_loader[1], structures, 'test_pred.csv')
