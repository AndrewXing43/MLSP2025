import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset import RadioMapDataset
from model import UNet
from utils import RMSELoss, save_checkpoint, custom_collate_fn, plot_loss_curve
from pathlib import Path
from tqdm import tqdm
import os
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False     

set_seed(42)

# ==== Config ====
data_root = Path("./")
inputs_dir = data_root / "inputs"
outputs_dir = data_root / "outputs"
sparse_dir = data_root / "mixed_samples_02"
positions_dir = data_root / "Positions"
los_dir = data_root / "losmap"
hit_dir = data_root / "hitmap"
acc_dir = data_root / "Tsummap"

batch_size = 4
epochs = 90
lr = 1e-4
val_ratio = 0.2
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# ==== Load dataset ====
full_dataset = RadioMapDataset(inputs_dir, outputs_dir, sparse_dir, positions_dir, los_dir = None, hit_dir = hit_dir, acc_dir=acc_dir, freq_filter=None)
val_size = int(len(full_dataset) * val_ratio)
train_size = len(full_dataset) - val_size
generator = torch.Generator().manual_seed(42)
train_set, val_set = random_split(full_dataset, [train_size, val_size], generator=generator)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

# ==== Initialize model ====
model = UNet(in_channels=5, out_channels=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = RMSELoss()
# # ==== Load checkpoint if exists ====
checkpoint_path = "checkpoints/best_model_base.pth"
if os.path.exists(checkpoint_path):
    print("Loading checkpoint...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
else:
    print("No checkpoint found. Starting from scratch.")

# ==== Training loop ====
best_val_loss = float('inf')
train_losses = []  
val_losses = []    

os.makedirs("checkpoints", exist_ok=True)

for epoch in range(1, epochs + 1):
    model.train()
    train_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
    for inputs, targets, masks in loop:
        inputs = inputs.to(device)
        targets = targets.to(device)
        masks = masks.to(device)

        preds = model(inputs)
        loss = criterion(preds, targets, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        loop.set_postfix(train_loss=loss.item())

    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)  #
    print(f"Epoch {epoch}: Train RMSE = {avg_train_loss:.4f}")

    # Validation
    val_squared_error_sum = 0.0
    val_pixel_count = 0

    with torch.no_grad():
        for inputs, targets, masks in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            masks = masks.to(device)

            preds = model(inputs)
            diff_sq = ((preds - targets) * 255) ** 2
            valid_mask = 1 - masks
            val_squared_error_sum += (diff_sq * valid_mask).sum().item()
            val_pixel_count += valid_mask.sum().item()

    global_val_rmse = (val_squared_error_sum / max(val_pixel_count, 1)) ** 0.5
    val_losses.append(global_val_rmse)
    print(f"Epoch {epoch}: Global Val RMSE = {global_val_rmse:.4f}")

    if global_val_rmse < best_val_loss:
        best_val_loss = global_val_rmse
        save_checkpoint(model, "checkpoints/best_model1.pth")
        print("Saved best model.")


# Visualize loss curves
plot_loss_curve(train_losses, val_losses, "checkpoints/loss_curve.png")