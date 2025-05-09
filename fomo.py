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

model = MobileNetV2Backbone()
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)

with torch.no_grad():
    output = model(dummy_input)

print("Output shape:", output.shape)