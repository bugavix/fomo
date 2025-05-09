import torch
import torch.nn as nn

class FOMOHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 1, filters: int = 32):
        super(FOMOHead, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, filters, kernel_size=1, stride=1),  # Keeps spatial size
            nn.ReLU(),
            nn.Conv2d(filters, num_classes, kernel_size=1)  # 1x1 conv for class logits
        )

    def forward(self, x):
        return self.head(x)
