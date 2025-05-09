import torch
import torch.nn as nn
from torchvision.models import MobileNet_V2_Weights
from ..backbone.mobilenetv2_backbone import MobileNetV2Backbone
from ..head.fomo_head import FOMOHead

class FOMOMobileNetV2(nn.Module):
    def __init__(self, filters: int = 32, num_classes: int = 1, weights: MobileNet_V2_Weights = None):
        super(FOMOMobileNetV2, self).__init__()
        self.backbone = MobileNetV2Backbone(weights=weights)
        self.head = FOMOHead(in_channels=filters, filters=filters, num_classes=num_classes)

    def forward(self, x):
        features = self.backbone(x)
        output = self.head(features)
        return output  # Shape: [B, num_classes, H/8, W/8]
