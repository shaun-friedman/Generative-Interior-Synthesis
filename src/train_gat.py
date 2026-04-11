import os
import shutil
import argparse
from time import perf_counter

import numpy as np
import zarr
import s3fs

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, GraphNorm
from torch_geometric.data import Data

from utils import download_and_extract_state_dict

CLASS_WEIGHTS = torch.tensor([
    0.0018, 0.0018, 0.0018, 0.0015, 0.1089, 
    0.0364, 0.0095, 0.0014, 0.1662,
    0.0017, 0.4894, 0.0426, 0.1370], dtype=torch.float)

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

        edge_attr = self.store['edge_attr'][i]      # [2, E]
        edge_index = self.store['edge_index'][i] - 5   # [2, E]
        edge_totals = self.store['edge_totals'][i]
        node_features = self.store['node_features'][i]
        node_totals = self.store['node_totals'][i]
        node_labels = self.store['node_labels'][i]

        n = int(node_totals[0])
        e = int(edge_totals[0])

        # --- slice valid ---
        x = torch.from_numpy(node_features[:n]).float()
        x[:, :2] /= 256.0   # scaling
        x[:, 2]  /= 10000.0
        
        y = torch.from_numpy(node_labels[:n]).long()

        ei = torch.from_numpy(edge_index[:, :e]).long()
        ea = torch.from_numpy(edge_attr[:, :e]).float()

        # --- make bidirectional ---
        if e > 0:
            ei = torch.cat([ei, ei.flip(0)], dim=1)
            ea = torch.cat([ea, ea], dim=1)

        # --- final shape for PyG ---
        edge_attr = ea.T.contiguous()  # [E, 2]

        # --- data validation ---
        assert ei.numel() == 0 or ei.max() < n, f"Edge index out of bounds: {ei.max()} >= {n}"
        assert ei.numel() == 0 or ei.min() >= 0, f"Negative edge index: {ei.min()}"
        assert y.min() >= 0 and y.max() < 13

        return Data(
            x=x,
            edge_index=ei,
            edge_attr=edge_attr,
            y=y
        )
        

# ---------------------------------------------------------------------------
# GAT Model
# ---------------------------------------------------------------------------
class FloorplanGNN(nn.Module):
    """
    Node-classification GNN for floorplan graphs.

    Architecture
    ─────────────
    GATConv layer 1  :  3  → hidden (multi-head attention)
    GATConv layer 2  :  hidden*heads → hidden
    Linear classifier:  hidden → num_classes

    Edge attributes (strength, type) are passed to each GATConv so the
    model can weight messages by wall vs door connectivity.
    """

    def __init__(
        self,
        num_node_features: int = 3,   # x, y, area
        num_edge_features: int = 2,   # strength, type
        num_classes:       int = 13,  # room categories
        hidden_dim:        int = 64,
        heads_1:           int = 4,
        heads_2:           int = 2,
        dropout:           float = 0.3,
    ):
        super().__init__()
        self.dropout = dropout

        # Layer 1: 3 → hidden*heads_1
        self.conv1 = GATConv(
            in_channels=num_node_features,
            out_channels=hidden_dim,
            heads=heads_1,
            edge_dim=num_edge_features,
            concat=True,
            dropout=dropout,
        )
        self.bn1 = GraphNorm(hidden_dim * heads_1)

        # Layer 2: hidden*heads_1 → hidden*heads_2
        self.conv2 = GATConv(
            in_channels=hidden_dim * heads_1,
            out_channels=hidden_dim,
            heads=heads_2,
            edge_dim=num_edge_features,
            concat=True,
            dropout=dropout,
        )
        self.bn2 = GraphNorm(hidden_dim * heads_2)

        # Classifier head
        self.classifier = torch.nn.Linear(hidden_dim * heads_2, num_classes)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Layer 1
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 2
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = self.bn2(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Node-level logits
        return self.classifier(x)   # [N, num_classes]

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
    
    model = FloorplanGNN(dropout=args.dropout).to(device)
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
        
    criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_loss = float("inf")
    
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(1, args.epochs + 1):
        epoch_start = perf_counter()

        # --- Train ---
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                logits = model(batch)
                loss = criterion(logits, batch.y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        acc = 0.0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)

                logits = model(batch)
                loss = criterion(logits, batch.y)
                preds = logits.argmax(dim=-1)
                acc += (preds == batch.y).float().mean().item()

                val_loss += loss.item()

        val_loss /= len(val_loader)
        acc /= len(val_loader)

        epoch_duration = perf_counter() - epoch_start

        print(f"Epoch {epoch:03d} | duration: {epoch_duration:.0f} | train loss: {train_loss:.4f} | val loss: {val_loss:.4f}| val acc: {acc:.4f}")

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
    parser.add_argument("--state-dict",    type=str,   default=None)
    parser.add_argument("--dropout",       type=float, default=0.3)
    
    # Data
    parser.add_argument("--train-idx", default=os.path.join(os.environ["SM_CHANNEL_INDICES"], "train_indices.npy"))
    parser.add_argument("--val-idx",   default=os.path.join(os.environ["SM_CHANNEL_INDICES"], "val_indices.npy"))
        
    args = parser.parse_args()

    train(args)
    os.makedirs("/opt/ml/model/code", exist_ok=True)
    shutil.copy("inference.py", "/opt/ml/model/code/inference.py")