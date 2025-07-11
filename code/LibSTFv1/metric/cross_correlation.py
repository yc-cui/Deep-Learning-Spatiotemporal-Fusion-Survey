import torch


def cross_correlation(pred, gt):
    N_spectral = pred.shape[1]

    # Rehsaping fused and reference data
    pred_reshaped = pred.view(N_spectral, -1)
    gt_reshaped = gt.view(N_spectral, -1)

    # Calculating mean value
    mean_fuse = torch.mean(pred_reshaped, 1).unsqueeze(1)
    mean_ref = torch.mean(gt_reshaped, 1).unsqueeze(1)
    CC = torch.sum((pred_reshaped - mean_fuse) * (gt_reshaped - mean_ref), 1) / \
        torch.sqrt(torch.sum((pred_reshaped - mean_fuse)**2, 1) * torch.sum((gt_reshaped - mean_ref)**2, 1))
    CC = torch.mean(CC)

    return CC
