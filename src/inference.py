from train import InteriorBoundsCNN

import torch
import numpy as np

def model_fn(model_dir):
    """Load model from the model_dir. Called once on container startup."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InteriorBoundsCNN()
    model.load_state_dict(
        torch.load(f"{model_dir}/model_best.pth", map_location=device)
    )
    model.to(device)
    model.eval()
    return model

def input_fn(request_body, content_type):
    """Deserialize the request body into a tensor. Called on each request."""
    if content_type == "application/x-npy":
        array = np.load(request_body)
        return torch.tensor(array, dtype=torch.float32)

def predict_fn(input_data, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_data = input_data.to(device)
    
    with torch.no_grad():
        output = model(input_data)
    return output.cpu().numpy()

def output_fn(prediction):
    return prediction