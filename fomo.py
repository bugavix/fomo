import torch
import torch.nn as nn
from torchvision.models import MobileNet_V2_Weights
from models_architecture.models.fomo_mobilenetv2 import FOMOMobileNetV2

model = FOMOMobileNetV2(num_classes=1, weights=MobileNet_V2_Weights.DEFAULT, filters=32)
model.eval()

from torchsummary import summary
summary(model, input_size=(3, 516, 516))

dummy_input = torch.randn(1, 3, 224, 224)

with torch.no_grad():
    output = model(dummy_input)

print("Output shape:", output.shape)