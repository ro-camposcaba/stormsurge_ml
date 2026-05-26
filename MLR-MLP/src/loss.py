import torch

#percentiles = torch.arange(0, 100.0001, 1, dtype=torch.float32) / 100.0

percentiles = torch.tensor([
    0.00, 0.01, 0.02, 0.03, 0.04, 0.05,
    0.06, 0.07, 0.08, 0.09, 0.10,
    0.20, 0.30, 0.40, 0.50,
    0.60, 0.70, 0.80, 0.90,
    0.95, 0.98, 0.99, 0.995, 0.999,
    1.00
], dtype=torch.float32)

def MADcLossSquared(z_true, z_pred):
    target_prc = torch.quantile(z_true, percentiles.to(z_true.device), dim=0)
    pred_prc = torch.quantile(z_pred, percentiles.to(z_pred.device), dim=0)
    mad_prc = torch.mean(torch.abs(target_prc - pred_prc))
    loss = torch.nanmean(torch.abs(z_true - z_pred)) + mad_prc
    return loss ** 2
