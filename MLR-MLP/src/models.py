import torch
import torch.nn as nn

# Linear regression model
class mlrModel(nn.Module):
    def __init__(self, input_size):
        super(mlrModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)

class mlpModel(torch.nn.Module):
    def __init__(self, device, inputSize, outputSize, hiddenLayerSizes=[120, 120]):
        super(mlpModel, self).__init__()
        self.inputSize = inputSize
        self.outputSize = outputSize

        self.inputLr = torch.nn.Linear(self.inputSize, hiddenLayerSizes[0], device=device)

        self.hiddenLayers = torch.nn.ModuleList()
        for ihl in range(len(hiddenLayerSizes) - 1):
            hl = torch.nn.Linear(hiddenLayerSizes[ihl], hiddenLayerSizes[ihl + 1], device=device)
            self.hiddenLayers.append(hl)

        self.outputLr = torch.nn.Linear(hiddenLayerSizes[-1], self.outputSize, device=device)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        out = self.inputLr(x)
        out = self.relu(out)
        for hlr in self.hiddenLayers:
            out = hlr(out)
            out = self.relu(out)
        out = self.outputLr(out)
        return out


def get_model(config, device):
    input_size = config["input_size"]
    output_size = config["output_size"]
    model_type = config["model_type"].lower()

    if model_type == "mlr":
        return mlrModel(input_size).to(device)

    elif model_type == "mlp":
        hidden_sizes = config.get("hidden_layer_sizes", [120, 120])
        return mlpModel(device,input_size, output_size, hidden_sizes).to(device)

    else:
        raise ValueError(f"Unknown model_type '{model_type}' in config.")

