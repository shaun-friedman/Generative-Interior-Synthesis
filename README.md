# Generative Interior Synthesis

A machine learning pipeline that learns the structural patterns of real-world floor plans and uses a **Graph Attention Network (GAT)** to generate novel, architecturally plausible interior layouts.

---

## Overview

This project treats floor plan generation as a graph-learning problem. Raw floor plan images are parsed into spatial masks (rooms, boundaries, doors, etc.), converted into graph representations, and used to train a GAT. The trained model can then synthesize new layouts by predicting graph structures that respect learned spatial relationships.

The project is structured as a sequential series of Jupyter notebooks, each corresponding to a stage of the ML pipeline.

---

## Repository Structure

```
Generative-Interior-Synthesis/
├── src/                        # Supporting Python modules
├── 1_Setup.ipynb               # Environment setup and dataset download
├── 2_Ingestion.ipynb           # Floor plan parsing and mask extraction
├── 3_EDA.ipynb                 # Exploratory data analysis and visualization
├── 4_Modeling.ipynb            # GAT model definition and training
├── 5_Testing.ipynb             # Inference, evaluation, and synthesis
├── gat_batch_pred.pt           # Saved GAT model weights
├── train_indices.npy           # Training split indices
├── val_indices.npy             # Validation split indices
├── batch_durations.npy         # Logged batch timing data
├── requirements.txt            # Python dependencies
└── LICENSE                     # MIT License
```

---

## Pipeline

### 1. Setup (`1_Setup.ipynb`)
Installs dependencies and downloads the floor plan dataset via `kagglehub`. Configure your Kaggle credentials before running.

### 2. Ingestion (`2_Ingestion.ipynb`)
Parses raw floor plan images into four binary spatial masks:

| Mask | Description |
|---|---|
| `inside_mask` | Interior area of the floor plan |
| `boundary_mask` | Walls and structural boundaries |
| `room_mask` | Individual room regions |
| `door_mask` | Door openings between rooms |

These masks are then converted into graph representations where nodes correspond to rooms and edges encode spatial adjacency.

### 3. EDA (`3_EDA.ipynb`)
Exploratory analysis of the parsed floor plans, including mask visualizations, room count distributions, adjacency statistics, and data quality checks.

### 4. Modeling (`4_Modeling.ipynb`)
Defines and trains a **Graph Attention Network (GAT)** using PyTorch Geometric. The model learns to predict spatial relationships between rooms from the graph-structured floor plan data. Trained weights are saved to `gat_batch_pred.pt`.

### 5. Testing (`5_Testing.ipynb`)
Loads the trained GAT and runs inference to synthesize new floor plan graphs. Evaluates output quality and visualizes generated layouts.

---

## Installation

```bash
git clone https://github.com/shaun-friedman/Generative-Interior-Synthesis.git
cd Generative-Interior-Synthesis
pip install -r requirements.txt
```

**Dependencies:**

- `torch-geometric` — Graph neural network framework
- `torchvision` — Image utilities
- `opencv-python` — Floor plan image processing
- `kagglehub` — Dataset download
- `zarr` / `s3fs` — Array storage and cloud I/O

> A CUDA-capable GPU is recommended for training.

---

## Dataset

The dataset is downloaded automatically in `1_Setup.ipynb` via `kagglehub`. You will need a [Kaggle account](https://www.kaggle.com) and API credentials (`~/.kaggle/kaggle.json`) configured before running setup.

---

## Usage

Run the notebooks in order:

```
1_Setup.ipynb → 2_Ingestion.ipynb → 3_EDA.ipynb → 4_Modeling.ipynb → 5_Testing.ipynb
```

A pre-trained model checkpoint (`gat_batch_pred.pt`) is included in the repository, so you can skip directly to `5_Testing.ipynb` to run inference without retraining.

---

## License

This project is licensed under the [MIT License](LICENSE).
