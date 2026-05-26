import torch
import torch.nn as nn


class lstmSimpleModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.output = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)        # (B, L, H)
        out = out[:, -1, :]          # last step
        out = self.output(out)       # (B, 1)
        return out


class lstmHybridModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_layer_size):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.lin = nn.Linear(input_size, hidden_layer_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(2 * hidden_layer_size, output_size)

    def forward(self, x):
        out_rnn, _ = self.lstm(x)
        out_rnn = out_rnn[:, -1, :]

        out_lin = self.relu(self.lin(x))
        out_lin = out_lin[:, -1, :]

        out = torch.cat((out_lin, out_rnn), dim=1)
        return self.output(out)


def get_model(config, device):
    input_size = config["input_size"]
    output_size = config["output_size"]
    model_type = config["model_type"].lower()

    if model_type == "lstm_simple":
        return lstmSimpleModel(input_size, config["hidden_layer_size"], output_size).to(device)

    elif model_type == "lstm_hybrid":
        return lstmHybridModel(input_size, output_size, config["hidden_layer_size"]).to(device)

    else:
        raise ValueError(f"Unknown model_type '{model_type}' in config.")

