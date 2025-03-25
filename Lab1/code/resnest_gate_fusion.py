import os
import numpy as np
import torch
import torch.nn as nn
import timm
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import torch.nn.functional as F

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


class NumericImageFolder(datasets.ImageFolder):
    """
    NumericImageFolder sorts class folder names based on numeric order
    and assigns them indices accordingly.
    """

    def find_classes(self, directory):
        classes = [
            d for d in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, d))
        ]
        classes.sort(key=lambda x: int(x))
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


def rand_bbox(size, lam):
    """
    Generate a random bounding box for CutMix.
    """
    width = size[3]
    height = size[2]
    cut_ratio = np.sqrt(1.0 - lam)
    cut_w = int(width * cut_ratio)
    cut_h = int(height * cut_ratio)
    cx = np.random.randint(width)
    cy = np.random.randint(height)
    bbx1 = np.clip(cx - cut_w // 2, 0, width)
    bby1 = np.clip(cy - cut_h // 2, 0, height)
    bbx2 = np.clip(cx + cut_w // 2, 0, width)
    bby2 = np.clip(cy + cut_h // 2, 0, height)
    return bbx1, bby1, bbx2, bby2


def cutmix_data(x, y, alpha=1.0):
    """
    Apply CutMix augmentation and return mixed images and adjusted labels.
    """
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)

    x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    lam = 1 - (
        (bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2])
    )
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Implements Label Smoothing for Cross Entropy.
    """

    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        log_prob = torch.log_softmax(x, dim=-1)
        n_classes = x.size(-1)
        true_dist = torch.zeros_like(x)
        true_dist.fill_(self.smoothing / (n_classes - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), 1 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * log_prob, dim=-1))


class SEModule(nn.Module):
    """
    Squeeze-and-Excitation module to perform channel-wise attention.
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, num_channels, _, _ = x.shape
        y = self.avg_pool(x).view(batch_size, num_channels)
        y = self.fc(y).view(batch_size, num_channels, 1, 1)
        return x * y


class GatedFusion(nn.Module):
    """
    Dynamically determine weighting for three feature branches (x2, x3, x4).
    1) Global average pooling => MLP => Sigmoid => gating weights
    2) Weighted sum of x2, x3, x4
    """

    def __init__(self, channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        hidden_dim = channels // 2
        self.mlp = nn.Sequential(
            nn.Linear(channels * 3, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x2, x3, x4):
        bsize, chans, _, _ = x2.shape
        x2_pool = self.avg_pool(x2).view(bsize, chans)
        x3_pool = self.avg_pool(x3).view(bsize, chans)
        x4_pool = self.avg_pool(x4).view(bsize, chans)

        x_cat = torch.cat([x2_pool, x3_pool, x4_pool], dim=1)
        alpha = self.mlp(x_cat)
        alpha2 = alpha[:, 0].view(bsize, 1, 1, 1)
        alpha3 = alpha[:, 1].view(bsize, 1, 1, 1)
        alpha4 = alpha[:, 2].view(bsize, 1, 1, 1)

        out = alpha2 * x2 + alpha3 * x3 + alpha4 * x4
        return out


class MultiScaleGatedSE_ResNeSt101e(nn.Module):
    """
    Multi-scale feature fusion model with GatedFusion, SE Module, and ResNeSt-101e.
    """

    def __init__(self, num_classes=100):
        super().__init__()
        backbone = timm.create_model(
            'resnest101e',
            pretrained=True,
            num_classes=0
        )

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.act = getattr(backbone, 'act', nn.ReLU(inplace=True))
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # Channel alignment: make layer2, layer3, layer4 outputs all 1024 channels
        self.conv2_1x1 = nn.Conv2d(512, 1024, kernel_size=1, bias=False)
        self.conv3_1x1 = nn.Conv2d(1024, 1024, kernel_size=1, bias=False)
        self.conv4_1x1 = nn.Conv2d(2048, 1024, kernel_size=1, bias=False)

        self.gated_fusion = GatedFusion(channels=1024)
        self.se = SEModule(1024, reduction=16)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x2 = self.layer2(x)   # shape: (B, 512, H/8, W/8)
        x3 = self.layer3(x2)  # shape: (B, 1024, H/16, W/16)
        x4 = self.layer4(x3)  # shape: (B, 2048, H/32, W/32)

        bsize, c2, h2, w2 = x2.shape
        x3_up = F.interpolate(
            x3, size=(h2, w2),
            mode='bilinear',
            align_corners=False
        )
        x4_up = F.interpolate(
            x4, size=(h2, w2),
            mode='bilinear',
            align_corners=False
        )

        x2_align = self.conv2_1x1(x2)     # shape: (B, 1024, H/8, W/8)
        x3_align = self.conv3_1x1(x3_up)  # shape: (B, 1024, H/8, W/8)
        x4_align = self.conv4_1x1(x4_up)  # shape: (B, 1024, H/8, W/8)

        x_gated = self.gated_fusion(x2_align, x3_align, x4_align)
        x_se = self.se(x_gated)
        out = self.classifier(x_se)
        return out


# Hyperparameters
NUM_CLASSES = 100
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 200
WEIGHT_DECAY = 1e-5
CUTMIX_ALPHA = 1.0
CUTMIX_PROB = 0.5

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "datasets", "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

# Transformations
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.2, contrast=0.2,
        saturation=0.2, hue=0.1
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# Build Dataset and DataLoader
train_dataset = NumericImageFolder(root=TRAIN_DIR, transform=train_transform)
val_dataset = NumericImageFolder(root=VAL_DIR, transform=val_transform)

'''
# check the index
print("Train dataset class_to_idx:")
for cls, idx in train_dataset.class_to_idx.items():
    print(f"class {cls} => {idx}")

print("\nValidation dataset class_to_idx:")
for cls, idx in val_dataset.class_to_idx.items():
    print(f"class {cls} => {idx}")
'''

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


def train_one_epoch(
    model, dataloader, criterion, optimizer,
    epoch_idx, total_epochs,
    cutmix_alpha=CUTMIX_ALPHA, cutmix_prob=CUTMIX_PROB
):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(dataloader, desc=f"Epoch [{epoch_idx+1}/{total_epochs}]", leave=False)

    for images, labels in pbar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        use_cutmix = (np.random.rand() < cutmix_prob)

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
    print(
        f"Epoch [{epoch_idx+1}/{total_epochs}] completed. "
        f"Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}"
    )
    return epoch_loss, epoch_acc


def validate_one_epoch(
    model, dataloader, criterion,
    epoch_idx, total_epochs
):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(dataloader, desc=f"Val Epoch [{epoch_idx+1}/{total_epochs}]", leave=False)

    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
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
    print(
        f"Validation Epoch [{epoch_idx+1}/{total_epochs}] completed. "
        f"Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}"
    )
    return epoch_loss, epoch_acc


def main():
    """
    Main training loop for MultiScaleGatedSE_ResNeSt101e with optional CutMix.
    """
    model = MultiScaleGatedSE_ResNeSt101e(num_classes=NUM_CLASSES).to(DEVICE)
    # criterion = nn.CrossEntropyLoss()
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.25,
        patience=5,
        verbose=True
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
            model,
            train_loader,
            criterion,
            optimizer,
            epoch_idx=epoch,
            total_epochs=EPOCHS,
            cutmix_alpha=CUTMIX_ALPHA,
            cutmix_prob=CUTMIX_PROB
        )
        val_loss, val_acc = validate_one_epoch(
            model,
            val_loader,
            criterion,
            epoch_idx=epoch,
            total_epochs=EPOCHS
        )
        print(
            f"[Epoch {epoch+1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f}, "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Acc: {val_acc:.4f}"
        )

        history_loss.append(train_loss)
        history_val_loss.append(val_loss)
        history_acc.append(train_acc)
        history_val_acc.append(val_acc)

        # Adjust LR based on validation accuracy
        scheduler.step(val_acc)

        # Save the best model
        is_better_acc = (val_acc > best_val_acc)
        is_same_acc_better_loss = (
            abs(val_acc - best_val_acc) < 1e-4 and val_loss < best_val_loss
        )
        if is_better_acc or is_same_acc_better_loss:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_weights = model.state_dict()
            filename = (
                f"ResNeSt101e_GatedSE1024_best_loss_cross_{val_loss:.4f}"
                f"_acc_{val_acc:.4f}.pt"
            )
            torch.save(best_weights, filename)
            print(f"New best model found! Saved to {filename}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s).")

        # Early stopping
        if epochs_no_improve >= 15:
            print("Early stopping triggered.")
            break

    # Plot loss
    plt.figure()
    plt.plot(history_loss, label="Train Loss", color="blue")
    plt.plot(history_val_loss, label="Val Loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs. Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot accuracy
    plt.figure()
    plt.plot(history_acc, label="Train Accuracy", color="green")
    plt.plot(history_val_acc, label="Val Accuracy", color="orange")
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
