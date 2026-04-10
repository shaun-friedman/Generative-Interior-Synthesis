import os
import shutil
import argparse
from time import perf_counter
from urllib.parse import urlparse

import numpy as np
import zarr
import s3fs

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from utils import download_and_extract_state_dict

class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, outputs, targets):
        # Apply sigmoid to convert logits to probabilities
        probs = torch.sigmoid(outputs)
        
        # Flatten tensors
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ZarrDataset(Dataset):
    def __init__(self, zarr_path, indices):
        self.zarr_path = zarr_path
        self.indices = indices
        #self.transforms = transforms
        print(f"DEBUG init zarr_path: {zarr_path}")

    def _open_store(self):
        if not hasattr(self, 'store'):
            self.store = zarr.open(self.zarr_path, mode='r')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        self._open_store()
        i = self.indices[idx] 

        inside_mask =   self.store['inside_masks'][i]
        boundary_mask = self.store['boundary_masks'][i]
        room_mask =     self.store['room_masks'][i]
        door_mask =     self.store['door_masks'][i]
        
        return {
            'inside_mask':   torch.from_numpy(inside_mask),
            'boundary_mask': torch.from_numpy(boundary_mask),
            'room_mask':     torch.from_numpy(room_mask),
            'door_mask':     torch.from_numpy(door_mask),
        }
        

# ---------------------------------------------------------------------------
# Interior Walls and Doors Model
# ---------------------------------------------------------------------------
class InteriorBoundsCNN(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        
        self.input_conv = self.conv_block(2, 3)
        
        # Backbone
        backbone = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        # Remove last 2 layers
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])

        # Encoder
        self.enc1 = self.conv_block(2, 32)
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
        x = self.input_conv(x)
        x = self.backbone(x)
        
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        x = nn.Conv2d(512, 2, kernel_size=1)(x)
        
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


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ResNetUNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        # ---- Input stem (2 → 3 channels) ----
        self.stem = nn.Conv2d(2, 3, kernel_size=3, padding=1)

        # ---- ResNet backbone ----
        backbone = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT if pretrained else None
        )

        # Extract layers
        self.layer0 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
        )  # 64 ch
        self.pool = backbone.maxpool

        self.layer1 = backbone.layer1  # 256 ch
        self.layer2 = backbone.layer2  # 512 ch
        self.layer3 = backbone.layer3  # 1024 ch
        self.layer4 = backbone.layer4  # 2048 ch

        # ---- Decoder ----
        self.up4 = UpBlock(2048, 1024, 1024)
        self.up3 = UpBlock(1024, 512, 512)
        self.up2 = UpBlock(512, 256, 256)
        self.up1 = UpBlock(256, 64, 128)

        self.final_up = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # ---- Output heads ----
        self.room_head = nn.Conv2d(64, 1, kernel_size=1)
        self.door_head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, inside_mask, boundary_mask):
        # Stack input
        x = torch.stack([inside_mask, boundary_mask], dim=1).float()

        # Stem
        x = self.stem(x)

        # ---- Encoder ----
        x0 = self.layer0(x)        # [B, 64, H/2, W/2]
        x1 = self.pool(x0)         # [B, 64, H/4, W/4]
        x1 = self.layer1(x1)       # [B, 256, H/4, W/4]
        x2 = self.layer2(x1)       # [B, 512, H/8, W/8]
        x3 = self.layer3(x2)       # [B, 1024, H/16, W/16]
        x4 = self.layer4(x3)       # [B, 2048, H/32, W/32]

        # ---- Decoder ----
        d4 = self.up4(x4, x3)
        d3 = self.up3(d4, x2)
        d2 = self.up2(d3, x1)
        d1 = self.up1(d2, x0)

        out = F.interpolate(d1, scale_factor=2, mode='bilinear', align_corners=False)
        out = self.final_up(out)

        return {
            "room_mask": self.room_head(out),
            "door_mask": self.door_head(out),
        }
# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

# ImageNet normalization — standard when using a pretrained ResNet backbone
IMAGENET_MEAN = 0.485
IMAGENET_STD  = 0.229

imagenet_transforms = transforms.Compose([
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------        
def train(args):
    print(f"DEBUG zarr verions:{zarr.__version__}")
    print(f"DEBUG s3fs verions:{s3fs.__version__}")
    model_dir = os.environ["SM_MODEL_DIR"]
    

    zarr_channel = os.environ["SM_CHANNEL_ZARR"]
    print(f"SM_CHANNEL_ZARR: {zarr_channel}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_idx = np.load(args.train_idx)
    val_idx = np.load(args.val_idx)
    
    zarr_path = os.path.join(zarr_channel, "data.zarr")
    print(f"DEBUG zarr_path:{zarr_path}")
    print("Exists?", os.path.exists(zarr_path))
        
    train_dataset = ZarrDataset(zarr_path, np.sort(train_idx))
    val_dataset   = ZarrDataset(zarr_path, val_idx)
    
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
    
    model = ResNetUNet().to(device)
    extract_dir = "./extracted_model"
    
    print("Checking state_dict arg...")
    print(args.state_dict)
    if args.state_dict:
        print("Checking for state dict..")
        state_dict_path = download_and_extract_state_dict(args.state_dict, extract_dir)
        
        if os.path.exists(state_dict_path):
            print(f"Loading state dict from {state_dict_path}")
            model.load_state_dict(torch.load(state_dict_path, map_location=device))
        else:
            print(f"Warning: model_best.pth not found in {args.state_dict}, training from scratch")
        
    criterion_1 = nn.BCEWithLogitsLoss()
    criterion_2 = BinaryDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_loss = float("inf")
    
    for epoch in range(1, args.epochs + 1):
        # --- Epoch Timing ---
        epoch_start = perf_counter()
        
        # --- Train ---
        model.train()
        train_loss = 0.0
        scaler = torch.cuda.amp.GradScaler()
        
        for batch in train_loader:
            inside_mask  = batch["inside_mask"].to(device).float()
            boundary_mask  = batch["boundary_mask"].to(device).float()
            room_mask  = batch["room_mask"].to(device).float().unsqueeze(1)
            door_mask  = batch["door_mask"].to(device).float().unsqueeze(1)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                predictions = model(inside_mask, boundary_mask)
                loss_1 = criterion_1(predictions['room_mask'], room_mask) \
                + criterion_1(predictions['door_mask'], door_mask)
                loss_2 = criterion_2(predictions['room_mask'], room_mask) \
                + criterion_2(predictions['door_mask'], door_mask)

            total_loss = loss_1 + loss_2
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += total_loss.item()

        train_loss /= len(train_loader)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inside_mask   = batch['inside_mask'].to(device).float()
                boundary_mask = batch['boundary_mask'].to(device).float()
                room_mask     = batch['room_mask'].to(device).float().unsqueeze(1)
                door_mask     = batch['door_mask'].to(device).float().unsqueeze(1)

                predictions = model(inside_mask, boundary_mask)
                loss_1 = criterion_1(predictions['room_mask'], room_mask) \
                + criterion_1(predictions['door_mask'], door_mask)
                loss_2 = criterion_2(predictions['room_mask'], room_mask) \
                + criterion_2(predictions['door_mask'], door_mask)
                
                total_loss = loss_1 + loss_2
                val_loss += total_loss.item()

        val_loss /= len(val_loader)
        epoch_end = perf_counter()
        epoch_duration = epoch_end - epoch_start
        print(f"Epoch {epoch:03d} | duration: {epoch_duration:.0f}| train loss: {train_loss:.4f} | val loss: {val_loss:.4f}")

        # --- Checkpoint ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(model_dir, "model_best.pth"))
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
    parser.add_argument("--state-dict", type=str, default=None)
    
    # Data
    parser.add_argument("--train-idx", default=os.path.join(os.environ["SM_CHANNEL_INDICES"], "train_indices.npy"))
    parser.add_argument("--val-idx",   default=os.path.join(os.environ["SM_CHANNEL_INDICES"], "val_indices.npy"))
        
    args = parser.parse_args()

    train(args)
    os.makedirs("/opt/ml/model/code", exist_ok=True)
    shutil.copy("inference.py", "/opt/ml/model/code/inference.py")