import torch
import torch.nn as nn

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.smooth_l1_loss = nn.SmoothL1Loss()
        self.mse_loss = nn.MSELoss()
        self.alpha = (
            alpha  # Weight for the SmoothL1Loss, (1-alpha) is the weight for MSELoss
        )

    def forward(self, outputs, targets):
        l1_loss = self.smooth_l1_loss(outputs, targets)
        mse_loss = self.mse_loss(outputs, targets)
        combined_loss = self.alpha * l1_loss + (1 - self.alpha) * mse_loss
        return combined_loss