import yaml
import torch
import importlib
from src.data_loader import load_data
from src.models import get_model
from src.train import train
import random
import numpy as np

# --------------------------------------------------
# Reproducibility settings
# --------------------------------------------------

GLOBAL_SEED = 42  

random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(GLOBAL_SEED)
    torch.cuda.manual_seed_all(GLOBAL_SEED)

# Force deterministic behavior (important for RNN/LSTM)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load config
with open("configs/config.yaml", 'r') as file:
    config = yaml.safe_load(file)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() and config['device'] == 'auto' else config['device'])
print("Using device:", device)

# Load data and determine input size dynamically
data = load_data(config, device)
# Set input size depending on model type
if len(data["XTrain"].shape) == 3:
    config["input_size"] = data["XTrain"].shape[2]
else:
    config["input_size"] = data["XTrain"].shape[1]

# Initialize model
model = get_model(config, device)

# Get loss function from config
loss_fn_name = config["loss_function"]
module_name = config.get("loss_module", "src.loss")  # defaults to custom module

# Dynamically import the loss module
loss_module = importlib.import_module(module_name)
loss_cls_or_fn = getattr(loss_module, loss_fn_name)

# Create an instance if necessary (torch.nn.MSELoss is a class, custom loss is a function)
if module_name == "torch.nn":
    criterion = loss_cls_or_fn()  # torch.nn.MSELoss()
else:
    criterion = loss_cls_or_fn    # e.g. MADcLossSquared

# Train
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    train(config, data, model, device, criterion)


