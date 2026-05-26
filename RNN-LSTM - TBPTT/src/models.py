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


class rnnSimpleModel(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size, num_layers=2):
        super().__init__()

        hidden_size = hidden_layer_sizes[0]
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        # x: (B, T, input_size)
        # hidden: (num_layers, B, hidden_size) or None

        out, hidden = self.rnn(x, hidden)   # out: (B, T, hidden_size)
        out = self.output(out)              # out: (B, T, output_size)

        return out, hidden


class rnnHybridModel(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size, num_layers=2):
        super().__init__()

        hidden_size = hidden_layer_sizes[0]
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.lin = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()

        self.output = nn.Linear(2 * hidden_size, output_size)

    def forward(self, x, hidden=None):
        # x: (B, T, input_size)
        # hidden: (num_layers, B, hidden_size) or None

        out_rnn, hidden = self.rnn(x, hidden)   # (B, T, hidden_size)
        out_lin = self.relu(self.lin(x))        # (B, T, hidden_size)

        out = torch.cat((out_lin, out_rnn), dim=2)  # (B, T, 2*hidden_size)
        out = self.output(out)                     # (B, T, output_size)

        return out, hidden


class lstmSimpleModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_layer_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.outputLr = nn.Linear(hidden_layer_size, output_size)

    # Updated section for the TBPTT approach
    def forward(self, x, hidden=None):
        out, hidden = self.lstm(x, hidden)      # (1, T, H)
        out = self.outputLr(out)   # (1, T, 1)
        return out, hidden


class lstmHybridModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_layer_size):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_layer_size,
            num_layers=1,
            batch_first=True
        )

        self.lin = nn.Linear(input_size, hidden_layer_size)
        self.relu = nn.ReLU()

        self.output = nn.Linear(2 * hidden_layer_size, output_size)

    def forward(self, x, hidden=None):
        # LSTM branch (temporal memory)
        out_rnn, hidden = self.lstm(x, hidden)      # (B, T, H)

        # Linear branch (instantaneous forcing)
        out_lin = self.relu(self.lin(x))            # (B, T, H)

        # Combine at EACH time step
        out = torch.cat((out_lin, out_rnn), dim=2)  # (B, T, 2H)

        # Final output per time step
        out = self.output(out)                      # (B, T, output_size)

        return out, hidden


def get_model(config, device):
    input_size = config["input_size"]
    output_size = config["output_size"]
    model_type = config["model_type"].lower()

    if model_type == "mlr":
        return mlrModel(input_size).to(device)

    elif model_type == "mlp":
        hidden_sizes = config.get("hidden_layer_sizes", [120, 120])
        return mlpModel(device,input_size, output_size, hidden_sizes).to(device)

    elif model_type == "lstm_simple":
        return lstmSimpleModel(input_size, config["hidden_layer_size"], output_size).to(device)

    elif model_type == "lstm_hybrid":
        return lstmHybridModel(input_size, output_size, config["hidden_layer_size"]).to(device)

    elif model_type == "rnn_simple":
        return rnnSimpleModel(input_size, [config["hidden_layer_size"]], output_size).to(device)

    elif model_type == "rnn_hybrid":
        return rnnHybridModel(input_size, [config["hidden_layer_size"]], output_size).to(device)

    else:
        raise ValueError(f"Unknown model_type '{model_type}' in config.")

