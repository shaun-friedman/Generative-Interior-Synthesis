import os
import shutil
import argparse

import numpy as np
import zarr
import s3fs

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ZarrDataset(Dataset):
    def __init__(self, zarr_path, indices):
        self.zarr_path = zarr_path
        self.indices = indices

    def _open_store(self):
        if not hasattr(self, 'store'):
            self.store = zarr.open(self.zarr_path, mode='r')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        self._open_store()
        i = self.indices[idx] 
        return {
            'inside_mask':   torch.from_numpy(self.store['inside_masks'][i]),
            'boundary_mask': torch.from_numpy(self.store['boundary_masks'][i]),
            'room_mask':     torch.from_numpy(self.store['room_masks'][i]),
            'door_mask':     torch.from_numpy(self.store['door_masks'][i]),
        }
        

# ---------------------------------------------------------------------------
# Interior Walls and Doors Model
# ---------------------------------------------------------------------------
class InteriorBoundsCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = self.conv_block(2, 32)   # 2 input channels: inside + boundary
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)

        # Decoder
        self.dec2 = self.conv_block(128 + 64, 64)
        self.dec1 = self.conv_block(64 + 32, 32)

        # Output heads — one per mask
        self.room_head = nn.Conv2d(32, 1, kernel_size=1)
        self.door_head = nn.Conv2d(32, 1, kernel_size=1)

        self.pool = nn.MaxPool2d(2)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, inside_mask, boundary_mask):
        x = torch.stack([inside_mask, boundary_mask], dim=1).float()

        # Encoder with skip connections
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Decoder with skip connections (U-Net style)
        d2 = self.dec2(torch.cat([F.interpolate(e3, scale_factor=2), e2], dim=1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, scale_factor=2), e1], dim=1))

        return {
            'room_mask': self.room_head(d1),
            'door_mask': self.door_head(d1),
        }
        
def train(args):
    model_dir = os.environ["SM_MODEL_DIR"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_idx = np.load(args.train_idx)
    val_idx = np.load(args.val_idx)
        
    train_dataset = ZarrDataset(args.zarr_path, np.sort(train_idx))
    val_dataset   = ZarrDataset(args.zarr_path, val_idx)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=min(16, os.cpu_count()),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(16, os.cpu_count()),
        pin_memory=True,
    )
    
    model     = InteriorBoundsCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):

        # --- Train ---
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            inside_mask  = batch["inside_mask"].to(device)
            boundary_mask  = batch["boundary_mask"].to(device)
            room_mask  = batch["room_mask"].to(device)
            door_mask  = batch["door_mask"].to(device)

            optimizer.zero_grad()
            predictions = model(inside_mask, boundary_mask)
            loss = criterion(predictions['room_mask'], room_mask) \
                + criterion(predictions['door_mask'], door_mask)
                
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inside_mask   = batch['inside_mask'].to(device).float()
                boundary_mask = batch['boundary_mask'].to(device).float()
                room_mask     = batch['room_mask'].to(device).float()
                door_mask     = batch['door_mask'].to(device).float()

                predictions = model(inside_mask, boundary_mask)
                loss = criterion(predictions['room_mask'], room_mask) \
                     + criterion(predictions['door_mask'], door_mask)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"Epoch {epoch:03d} | train loss: {train_loss:.4f} | val loss: {val_loss:.4f}")

        # --- Checkpoint ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pt"))
            print(f"  Saved new best model (val loss: {val_loss:.4f})")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Hyperparameters
    parser.add_argument("--epochs",        type=int,   default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size",    type=int,   default=32)
    
    # Data
    parser.add_argument("--zarr-path",     type=str, default=None)
    parser.add_argument("--train-idx",     default=os.path.join(os.environ["SM_CHANNEL_INDICES"], "train_indices.npy"))
    parser.add_argument("--val-idx",       default=os.path.join(os.environ["SM_CHANNEL_INDICES"], "val_indices.npy"))
    
    args = parser.parse_args()

    train(args)
    os.makedirs("/opt/ml/model/code", exist_ok=True)
    shutil.copy("inference.py", "/opt/ml/model/code/inference.py")