import os
import random
import torch
from torchvision.datasets import CocoDetection
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset
from models_architecture.models.fomo_mobilenetv2 import FOMOMobileNetV2

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Set to your local COCO path
coco_root = "coco"
coco_val = CocoDetection(
    root=os.path.join(coco_root, "val2017"),
    annFile=os.path.join(coco_root, "annotations", "instances_val2017.json"),
    transform=transform
)

subset_size = 1000
indices = list(range(len(coco_val)))
random.shuffle(indices)
subset = Subset(coco_val, indices[:subset_size])

# Split: 70% train, 15% val, 15% test
n = len(subset)
train_set = Subset(subset, list(range(0, int(0.7 * n))))
val_set   = Subset(subset, list(range(int(0.7 * n), int(0.85 * n))))
test_set  = Subset(subset, list(range(int(0.85 * n), n)))

def generate_heatmap_from_coco(boxes, labels, image_size=(224, 224), grid_size=8):
    heatmap = torch.zeros((1, image_size[0] // grid_size, image_size[1] // grid_size))
    for box, label in zip(boxes, labels):
        if label != 1:  # 1 = 'person' in COCO
            continue
        x, y, w, h = box
        x_center = x + w / 2
        y_center = y + h / 2
        grid_x = int((x_center / image_size[0]) * heatmap.shape[2])
        grid_y = int((y_center / image_size[1]) * heatmap.shape[1])
        if 0 <= grid_x < heatmap.shape[2] and 0 <= grid_y < heatmap.shape[1]:
            heatmap[0, grid_y, grid_x] = 1.0
    return heatmap

class FOMOCocoDataset(Dataset):
    def __init__(self, coco_subset):
        self.data = coco_subset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, anns = self.data[idx]
        boxes = [obj['bbox'] for obj in anns]
        labels = [obj['category_id'] for obj in anns]
        heatmap = generate_heatmap_from_coco(boxes, labels)
        return img, heatmap

train_loader = DataLoader(FOMOCocoDataset(train_set), batch_size=8, shuffle=True)
val_loader   = DataLoader(FOMOCocoDataset(val_set), batch_size=8)
test_loader  = DataLoader(FOMOCocoDataset(test_set), batch_size=8)

model = FOMOMobileNetV2(num_classes=1)  # defined earlier
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.BCEWithLogitsLoss()

# Initialize variables
epoch_number = 100
best_val_loss = float('inf')  # Start with a very large number
patience = 5  # How many epochs to wait before early stopping
counter = 0  # Counter to track patience

# Training loop (with validation loss tracking)
for epoch in range(epoch_number):  # Running epochs
    model.train()
    total_train_loss = 0
    
    for imgs, heatmaps in train_loader:
        preds = model(imgs)
        loss = criterion(preds, heatmaps)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()

    # Validation
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for imgs, heatmaps in val_loader:
            preds = model(imgs)
            loss = criterion(preds, heatmaps)
            total_val_loss += loss.item()

    print(f"Epoch {epoch+1} | Train Loss: {total_train_loss:.4f} | Val Loss: {total_val_loss:.4f}")

    # Save the best model (if validation loss decreases)
    if total_val_loss < best_val_loss:
        print(f"Validation loss decreased: saving model...")
        best_val_loss = total_val_loss
        torch.save(model.state_dict(), "fomo_mobilenetv2_best.pth")
        counter = 0  # Reset counter if improvement
    else:
        counter += 1
    
    # Early stopping: stop training if no improvement for 'patience' epochs
    if counter >= patience:
        print("Early stopping: no improvement in validation loss for 5 epochs")
        break

def evaluate_fomo(model, dataloader, threshold=0.5):
    model.eval()
    TP = FP = TN = FN = 0

    with torch.no_grad():
        for imgs, targets in dataloader:
            outputs = torch.sigmoid(model(imgs))
            preds = (outputs > threshold).float()

            TP += ((preds == 1) & (targets == 1)).sum().item()
            FP += ((preds == 1) & (targets == 0)).sum().item()
            TN += ((preds == 0) & (targets == 0)).sum().item()
            FN += ((preds == 0) & (targets == 1)).sum().item()

    precision = TP / (TP + FP + 1e-6)
    recall    = TP / (TP + FN + 1e-6)
    f1_score  = 2 * (precision * recall) / (precision + recall + 1e-6)
    accuracy  = (TP + TN) / (TP + TN + FP + FN)

    print("\n🎯 Evaluation Metrics:")
    print(f"Accuracy : {accuracy*100:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1_score:.4f}")
    print(f"(TP={TP}, FP={FP}, TN={TN}, FN={FN})")

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1_score}

# Final evaluation
evaluate_fomo(model, test_loader)