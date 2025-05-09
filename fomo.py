import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

class MobileNetV2Backbone(nn.Module):
    def __init__(self, output_stride=8):
        super(MobileNetV2Backbone, self).__init__()
        full_model = mobilenet_v2(weights = MobileNet_V2_Weights.DEFAULT)
        
        # MobileNetV2 uses stride-2 downsampling at various points
        # Stopping at 1/8th resolution = after layer 6 of inverted residual blocks

        # We keep all layers up to the point where resolution is downsampled by 8
        self.features = nn.Sequential(
            full_model.features[0],   # ConvBNReLU, stride=2  112x112
            full_model.features[1],   # InvertedResidual, stride=1
            full_model.features[2],   # InvertedResidual, stride=2  56x56
            full_model.features[3],   # InvertedResidual, stride=1
            full_model.features[4],   # InvertedResidual, stride=2  28x28
            full_model.features[5],   # InvertedResidual, stride=1
            full_model.features[6],   # InvertedResidual, stride=1
        )

    def forward(self, x):
        return self.features(x)

class FOMOHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super(FOMOHead, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=1, stride=1),  # Keeps spatial size
            nn.ReLU(),
            nn.Conv2d(32, num_classes, kernel_size=1)  # 1x1 conv for class logits
        )

    def forward(self, x):
        return self.head(x)

# Final Combined Model
class FOMOMobileNetV2(nn.Module):
    def __init__(self, num_classes: int):
        super(FOMOMobileNetV2, self).__init__()
        self.backbone = MobileNetV2Backbone()
        self.head = FOMOHead(32, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        output = self.head(features)
        return output  # Shape: [B, num_classes, H/8, W/8]

model = FOMOMobileNetV2(num_classes=1)
model.eval()

from torchsummary import summary
summary(model, input_size=(3, 224, 224))

dummy_input = torch.randn(1, 3, 224, 224)

with torch.no_grad():
    output = model(dummy_input)

print("Output shape:", output.shape)