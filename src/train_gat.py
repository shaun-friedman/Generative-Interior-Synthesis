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
from torch.utils.data import DataLoader, Dataset
from torch_geometric.nn import GATConv, BatchNorm

from utils import download_and_extract_state_dict

CLASS_WEIGHTS = torch.tensor([0.147306, 0.146718, 0.141799, 0.177072, 
                 0.002392, 0.007162, 0.027323, 0.182312, 
                 0.001568, 0.157803, 0.000532, 0.006110, 
                 0.001902], dtype=torch.float)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ZarrDataset(Dataset):
    def __init__(self, zarr_path, indices):
        self.zarr_path = zarr_path
        self.indices = indices
        print(f"DEBUG init zarr_path: {zarr_path}")

    def _open_store(self):
        if not hasattr(self, 'store'):
            self.store = zarr.open(self.zarr_path, mode='r')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        self._open_store()
        i = self.indices[idx] 

        edge_attr =   self.store['edge_attr'][i]
        edge_index = self.store['edge_index'][i] - 5 # zero index all nodes
        edge_totals =     self.store['edge_totals'][i]
        node_features =     self.store['node_features'][i]
        node_totals =     self.store['node_totals'][i]
        node_labels =     self.store['node_labels'][i]
        
        return {
            'edge_attr':   torch.from_numpy(edge_attr),
            'edge_index': torch.from_numpy(edge_index),
            'edge_totals':     torch.from_numpy(edge_totals),
            'node_features':     torch.from_numpy(node_features),
            'node_totals':     torch.from_numpy(node_totals),
            'node_labels':     torch.from_numpy(node_labels),
        }
        

# ---------------------------------------------------------------------------
# GAT Model
# ---------------------------------------------------------------------------
class FloorplanGNN(torch.nn.Module):
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
        self.bn1 = BatchNorm(hidden_dim * heads_1)

        # Layer 2: hidden*heads_1 → hidden*heads_2
        self.conv2 = GATConv(
            in_channels=hidden_dim * heads_1,
            out_channels=hidden_dim,
            heads=heads_2,
            edge_dim=num_edge_features,
            concat=True,
            dropout=dropout,
        )
        self.bn2 = BatchNorm(hidden_dim * heads_2)

        # Classifier head
        self.classifier = torch.nn.Linear(hidden_dim * heads_2, num_classes)

    def forward(self, data):
        x, edge_index, edge_attr = data["x"], data["edge_index"], data["edge_attr"]

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
    
    model = FloorplanGNN().to(device)
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
        # --- Epoch Timing ---
        epoch_start = perf_counter()
        
        # --- Train ---
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            node_totals = batch["node_totals"].squeeze(-1)   # [B]
            edge_totals = batch["edge_totals"].squeeze(-1)   # [B]

            # Unpad and concatenate nodes across the batch
            node_features_list, node_labels_list = [], []
            edge_index_list, edge_attr_list = [], []
            node_offset = 0

            for i in range(len(node_totals)):
                n = node_totals[i].item()
                e = edge_totals[i].item()

                node_features_list.append(batch["node_features"][i, :n, :])   # [n, 3]
                node_labels_list.append(batch["node_labels"][i, :n])           # [n]

                ei = batch["edge_index"][i, :, :e].long()                      # [2, e]

                # Make bidirectional
                ei_bidir = torch.cat([ei, ei.flip(0)], dim=1)           # [2, 2e]
                ea = batch["edge_attr"][i, :, :e]
                ea_bidir = torch.cat([ea, ea], dim=1)                    # [2, 2e] — same attrs both directions

                ei_bidir = ei_bidir + node_offset
                edge_index_list.append(ei_bidir)
                edge_attr_list.append(ea_bidir)           # [2, e]

                node_offset += n

            x           = torch.cat(node_features_list, dim=0).to(device).float()   # [N_total, 3]
            node_labels = torch.cat(node_labels_list,   dim=0).to(device).long()    # [N_total]
            edge_index  = torch.cat(edge_index_list,    dim=1).to(device).long()    # [2, E_total]
            edge_attr   = torch.cat(edge_attr_list,     dim=1).to(device).float().T # [E_total, 2]

            assert node_labels.min() >= 0, \
                f"Negative label found: min={node_labels.min()}, likely unmatched padding sentinel"
            assert node_labels.max() < 13, \
                f"Label {node_labels.max()} exceeds num_classes=13"
                
            data = {"x": x, "edge_index": edge_index, "edge_attr": edge_attr}

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits = model(data)
                loss   = F.cross_entropy(logits, node_labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                node_totals = batch["node_totals"].squeeze(-1)   # [B]
                edge_totals = batch["edge_totals"].squeeze(-1)   # [B]

                # Unpad and concatenate nodes across the batch
                node_features_list, node_labels_list = [], []
                edge_index_list, edge_attr_list = [], []
                node_offset = 0

                for i in range(len(node_totals)):
                    n = node_totals[i].item()
                    e = edge_totals[i].item()

                    node_features_list.append(batch["node_features"][i, :n, :])   # [n, 3]
                    node_labels_list.append(batch["node_labels"][i, :n])           # [n]

                    ei = batch["edge_index"][i, :, :e].long()                      # [2, e]
                    ei = ei + node_offset                                          # global offset
                    edge_index_list.append(ei)
                    edge_attr_list.append(batch["edge_attr"][i, :, :e])            # [2, e]

                    node_offset += n

                x           = torch.cat(node_features_list, dim=0).to(device).float()   # [N_total, 3]
                node_labels = torch.cat(node_labels_list,   dim=0).to(device).long()    # [N_total]
                edge_index  = torch.cat(edge_index_list,    dim=1).to(device).long()    # [2, E_total]
                edge_attr   = torch.cat(edge_attr_list,     dim=1).to(device).float().T # [E_total, 2]

                data = {"x": x, "edge_index": edge_index, "edge_attr": edge_attr}

                logits = model(data)
                loss = criterion(logits, node_labels)

                val_loss += loss.item()

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
    parser.add_argument("--state-dict",    type=str,   default=None)
    
    # Data
    parser.add_argument("--train-idx", default=os.path.join(os.environ["SM_CHANNEL_INDICES"], "train_indices.npy"))
    parser.add_argument("--val-idx",   default=os.path.join(os.environ["SM_CHANNEL_INDICES"], "val_indices.npy"))
        
    args = parser.parse_args()

    train(args)
    os.makedirs("/opt/ml/model/code", exist_ok=True)
    shutil.copy("inference.py", "/opt/ml/model/code/inference.py")