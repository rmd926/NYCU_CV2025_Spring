import os
import random
import numpy as np
import torch
import torch.nn as nn
import timm  # Used for creating the resnest101e model
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# Set environment variables and seeds
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONHASHSEED"] = "42"

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Custom ImageFolder: Sort folders numerically
class NumericImageFolder(datasets.ImageFolder):
    def find_classes(self, directory):
        classes = [
            d for d in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, d))
        ]
        classes.sort(key=lambda x: int(x))
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

# CutMix Functions
def rand_bbox(size, lam):
    W = size[3]
    H = size[2]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) /
               (x.size()[-1] * x.size()[-2]))
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam

# Label Smoothing CrossEntropy Loss
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        log_prob = torch.log_softmax(x, dim=-1)
        n_classes = x.size(-1)
        true_dist = torch.zeros_like(x)
        true_dist.fill_(self.smoothing / (n_classes - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), 1 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * log_prob, dim=-1))

# Hyperparameters and Data Paths
NUM_CLASSES = 100
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 100
WEIGHT_DECAY = 1e-5

CUTMIX_ALPHA = 1.0
CUTMIX_PROB = 0.5

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "datasets", "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

# Image Preprocessing
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                           saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Create Dataset and DataLoader
train_dataset = NumericImageFolder(root=TRAIN_DIR, transform=train_transform)
val_dataset = NumericImageFolder(root=VAL_DIR, transform=val_transform)

# print("Train dataset class_to_idx:")
# for cls, idx in train_dataset.class_to_idx.items():
#     print(f"class {cls} ; Reality: {idx}")
# print("\nValidation dataset class_to_idx:")
# for cls, idx in val_dataset.class_to_idx.items():
#     print(f"class {cls} ; Reality: {idx}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Training and Validation Process
def train_one_epoch(model, dataloader, criterion, optimizer, epoch_idx,
                    total_epochs, cutmix_alpha=CUTMIX_ALPHA, cutmix_prob=CUTMIX_PROB):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(dataloader, desc=f"Epoch [{epoch_idx+1}/{total_epochs}]", leave=False)
    for images, labels in pbar:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        use_cutmix = np.random.rand() < cutmix_prob

        if use_cutmix:
            images, y_a, y_b, lam = cutmix_data(images, labels, alpha=cutmix_alpha)
            outputs = model(images)
            loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        if use_cutmix:
            correct += (preds == y_a).sum().item()
        else:
            correct += (preds == labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{(correct / total):.4f}"
        })
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    print(f"Epoch [{epoch_idx+1}/{total_epochs}] completed. Loss: {epoch_loss:.4f}, "
          f"Acc: {epoch_acc:.4f}")
    return epoch_loss, epoch_acc


def validate_one_epoch(model, dataloader, criterion, epoch_idx, total_epochs):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(dataloader, desc=f"Val Epoch [{epoch_idx+1}/{total_epochs}]", leave=False)
    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix({
                "val_loss": f"{loss.item():.4f}",
                "val_acc": f"{(correct / total):.4f}"
            })
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    print(f"Validation Epoch [{epoch_idx+1}/{total_epochs}] completed. Loss: {epoch_loss:.4f}, "
          f"Acc: {epoch_acc:.4f}")
    return epoch_loss, epoch_acc


def main():
    # Create the resnest101e model using timm, with pretrained weights and set number of classes
    model = timm.create_model('resnest101e', pretrained=True, num_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    #criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.25, patience=5, verbose=True
    )

    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_weights = None
    epochs_no_improve = 0

    history_loss = []
    history_val_loss = []
    history_acc = []
    history_val_acc = []

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer,
            epoch_idx=epoch, total_epochs=EPOCHS,
            cutmix_alpha=CUTMIX_ALPHA, cutmix_prob=CUTMIX_PROB
        )
        val_loss, val_acc = validate_one_epoch(
            model, val_loader, criterion, epoch_idx=epoch, total_epochs=EPOCHS
        )
        print(f"[Epoch {epoch+1}/{EPOCHS}] Train Loss: {train_loss:.4f}, "
              f"Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, "
              f"Val Acc: {val_acc:.4f}")

        history_loss.append(train_loss)
        history_val_loss.append(val_loss)
        history_acc.append(train_acc)
        history_val_acc.append(val_acc)

        scheduler.step(val_acc)

        if (val_acc > best_val_acc) or (
            abs(val_acc - best_val_acc) < 1e-4 and val_loss < best_val_loss
        ):
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_weights = model.state_dict()
            filename = f"ResNest101e_best_loss_{val_loss:.4f}_acc_{val_acc:.4f}.pt"
            torch.save(best_weights, filename)
            print(f"New best model found! Saved to {filename}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= 15:
            print("Early stopping triggered.")
            break

    plt.figure()
    plt.plot(history_loss, label="Train Loss", color="blue")
    plt.plot(history_val_loss, label="Validation Loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs. Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(history_acc, label="Train Accuracy", color="green")
    plt.plot(history_val_acc, label="Validation Accuracy", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs. Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

    if best_weights is not None:
        model.load_state_dict(best_weights)
    print("Training finished.")


if __name__ == "__main__":
    main()
