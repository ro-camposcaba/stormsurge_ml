import torch

percentiles = torch.arange(0, 100.0001, 1, dtype=torch.float32) / 100.0

def MADcLossSquared(z_true, z_pred):
    target_prc = torch.quantile(z_true, percentiles.to(z_true.device), dim=0)
    pred_prc = torch.quantile(z_pred, percentiles.to(z_pred.device), dim=0)
    mad_prc = torch.mean(torch.abs(target_prc - pred_prc))
    loss = torch.nanmean(torch.abs(z_true - z_pred)) + mad_prc
    return loss ** 2
