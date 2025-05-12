import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from models_architecture.models.fomo_mobilenetv2 import FOMOMobileNetV2

# Instantiate model
model = FOMOMobileNetV2(num_classes=1)
model.load_state_dict(torch.load("fomo_mobilenetv2_best.pth", map_location="cpu"))
model.eval()

# Load and preprocess image
image = Image.open("example-1.jpg").convert("RGB")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])
input_tensor = transform(image).unsqueeze(0)  # Shape: (1, 3, 224, 224)

# Inference
with torch.no_grad():
    output = model(input_tensor)
    heatmap = torch.sigmoid(output[0, 0])  # Shape: (H, W)
    # heatmap = (output[0, 0] > 0).float()

# Show grayscale heatmap
plt.imshow(heatmap.numpy(), cmap="gray")
plt.title("FOMO Heatmap")
plt.axis("off")
plt.show()