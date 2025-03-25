import os
import csv
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm

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
NUM_CLASSES = 100
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "datasets", "data")
TEST_DIR = os.path.join(DATA_DIR, "test")


# Testing Dataset Definition
class TestImageDataset(Dataset):
    """
    Custom dataset for test images.
    Each image is sorted by filename, transformed, and returned.
    """

    def __init__(self, test_dir, transform=None):
        self.image_names = sorted(os.listdir(test_dir))
        self.image_paths = [
            os.path.join(test_dir, fname) for fname in self.image_names
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        base_name = os.path.splitext(self.image_names[idx])[0]
        return image, base_name


test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
test_dataset = TestImageDataset(test_dir=TEST_DIR, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


class SEModule(nn.Module):
    """
    Squeeze-and-Excitation (SE) module for channel-wise attention.
    """

    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        y = self.avg_pool(x).view(B, C)
        y = self.fc(y).view(B, C, 1, 1)
        return x * y


class GatedFusion(nn.Module):
    """
    Gated fusion module that fuses three feature branches with the same channel
    dimension.
    1) Global average pooling -> MLP -> Sigmoid for alpha2, alpha3, alpha4.
    2) Weighted sum of x2, x3, and x4.
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
        bsz, chans, _, _ = x2.shape
        x2_pool = self.avg_pool(x2).view(bsz, chans)
        x3_pool = self.avg_pool(x3).view(bsz, chans)
        x4_pool = self.avg_pool(x4).view(bsz, chans)
        x_cat = torch.cat([x2_pool, x3_pool, x4_pool], dim=1)
        alpha = self.mlp(x_cat)
        alpha2 = alpha[:, 0].view(bsz, 1, 1, 1)
        alpha3 = alpha[:, 1].view(bsz, 1, 1, 1)
        alpha4 = alpha[:, 2].view(bsz, 1, 1, 1)
        out = alpha2 * x2 + alpha3 * x3 + alpha4 * x4
        return out


# Gate model: uses gated fusion
class MultiScaleGatedSE_ResNeSt101e(nn.Module):
    """
    Multi-scale feature fusion with ResNeSt-101e backbone, gated fusion, and an SE
    module.
    1) Extract layer2, layer3, and layer4 features.
    2) Project each branch to 1024 channels.
    3) Fuse them with the GatedFusion module.
    4) Refine with SE.
    5) Classify.
    """

    def __init__(self, num_classes=100):
        super().__init__()
        backbone = timm.create_model("resnest101e", pretrained=True, num_classes=0)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.act = getattr(backbone, "act", nn.ReLU(inplace=True))
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # Channel alignment to 1024 for layer2, layer3, and layer4 outputs
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
        x2 = self.layer2(x)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        bsz, c2, h2, w2 = x2.shape
        x3_up = F.interpolate(x3, size=(h2, w2), mode="bilinear",
                              align_corners=False)
        x4_up = F.interpolate(x4, size=(h2, w2), mode="bilinear",
                              align_corners=False)
        x2_align = self.conv2_1x1(x2)
        x3_align = self.conv3_1x1(x3_up)
        x4_align = self.conv4_1x1(x4_up)
        x_gated = self.gated_fusion(x2_align, x3_align, x4_align)
        x_se = self.se(x_gated)
        return self.classifier(x_se)


# Pyramid model: uses concatenation fusion 
class MultiScaleResNeStSE(nn.Module):
    """
    Multi-scale fusion with ResNeSt-101e backbone and SE module.
    1) Extract layer1, layer2, layer3, and layer4 features.
    2) Upsample layer3 and layer4 to match the spatial size of layer2.
    3) Concatenate the features (channels: 512 + 1024 + 2048 = 3584).
    4) Fuse the concatenated features with a 1x1 convolution, followed by the SE module.
    5) Classify.
    """

    def __init__(self, num_classes=100):
        super().__init__()
        backbone = timm.create_model("resnest101e", pretrained=True, num_classes=0)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.act = getattr(backbone, "act", nn.ReLU(inplace=True))
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2  # (B, 512, H/8, W/8)
        self.layer3 = backbone.layer3  # (B, 1024, H/16, W/16)
        self.layer4 = backbone.layer4  # (B, 2048, H/32, W/32)

        self.fuse_conv = nn.Conv2d(3584, 2048, kernel_size=1, bias=False)
        self.se = SEModule(2048, reduction=16)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x2 = self.layer2(x)  # (B, 512, H/8, W/8)
        x3 = self.layer3(x2)  # (B, 1024, H/16, W/16)
        x4 = self.layer4(x3)  # (B, 2048, H/32, W/32)
        B, c2, h2, w2 = x2.shape
        x3_up = F.interpolate(x3, size=(h2, w2), mode="bilinear",
                              align_corners=False)
        x4_up = F.interpolate(x4, size=(h2, w2), mode="bilinear",
                              align_corners=False)
        x_cat = torch.cat([x2, x3_up, x4_up], dim=1)
        x_fused = self.fuse_conv(x_cat)
        x_se = self.se(x_fused)
        return self.classifier(x_se)


# Plain ResNeSt model: using the default timm model without additional fusion/SE
def create_plain_resnest(num_classes=100):
    """
    Creates a plain resnest101e model from timm.
    """
    model = timm.create_model("resnest101e", pretrained=True,
                              num_classes=num_classes)
    return model.to(DEVICE)


# Inference Function
def inference(model, dataloader, output_csv="prediction.csv"):
    """
    Performs inference on the test set and saves the results in a CSV file.
    """
    model.eval()
    predictions = []
    with torch.no_grad():
        for images, filenames in tqdm(dataloader, desc="Inference", leave=False):
            images = images.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            preds = preds.cpu().numpy().tolist()
            for fname, p in zip(filenames, preds):
                predictions.append((fname, p))
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "pred_label"])
        for fname, label in predictions:
            writer.writerow([fname, label])
    print(f"Inference completed. Predictions saved to {output_csv}")


# Main Function 
def main():
    parser = argparse.ArgumentParser(
        description="Inference script with selectable model (Gate, Pyramid, or plain ResNeSt)"
    )
    parser.add_argument(
        "--model", type=str, choices=["gate", "pyramid", "resnest"],
        default="pyramid",
        help=("Choose the model type: 'gate' for MultiScaleGatedSE_ResNeSt101e, "
              "'pyramid' for MultiScaleResNeStSE, 'resnest' for plain resnest101e "
              "(default: pyramid)")
    )
    parser.add_argument(
        "--weights", type=str, default="",
        help="Path to the trained weights file (if available)"
    )
    args = parser.parse_args()

    if args.model == "gate":
        print("Using the Gate model.")
        model = MultiScaleGatedSE_ResNeSt101e(num_classes=NUM_CLASSES).to(DEVICE)
    elif args.model == "pyramid":
        print("Using the Pyramid model.")
        model = MultiScaleResNeStSE(num_classes=NUM_CLASSES).to(DEVICE)
    elif args.model == "resnest":
        print("Using the plain ResNeSt model.")
        model = create_plain_resnest(num_classes=NUM_CLASSES)
    else:
        raise ValueError("Unknown model type.")

    if args.weights and os.path.exists(args.weights):
        model.load_state_dict(torch.load(args.weights, map_location=DEVICE))
        print(f"Loaded model weights from {args.weights}")
    else:
        print("Warning: Weights file not found or not provided. Using random weights.")

    inference(model, test_loader, output_csv="prediction.csv")
    print("Inference finished. You can now submit the CSV file.")


if __name__ == "__main__":
    main()
