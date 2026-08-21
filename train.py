import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import copy
import time
import torch

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

MODEL_CHECKPOINTS = {
    "EfficientNetB0": "efficientnetb0_gbc.pth",
    "EfficientNetB1": "efficientnetb1_gbc.pth",
    "EfficientNetB2": "efficientnetb2_gbc.pth",
    "EfficientNetB3": "efficientnetb3_gbc.pth",
    "EfficientNetB4": "efficientnetb4_gbc.pth",
    "ResNet18": "resnet18_gbc.pth",
    "ResNet50": "resnet50_gbc.pth",
    "MobileNetV2": "mobilenetv2_gbc.pth",
    "DenseNet121": "densenet121_gbc.pth",
}

def build_model(model_name, num_classes=5):
    if model_name == "EfficientNetB0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif model_name == "EfficientNetB1":
        m = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif model_name == "EfficientNetB2":
        m = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif model_name == "EfficientNetB3":
        m = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif model_name == "EfficientNetB4":
        m = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif model_name == "ResNet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif model_name == "ResNet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif model_name == "MobileNetV2":
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif model_name == "DenseNet121":
        m = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return m

def train_single_model(model_name, dataloaders, dataset_sizes, device, num_epochs=10, lr=1e-4):
    print(f"\n==========================================")
    print(f"Starting Training for: {model_name}")
    print(f"==========================================")

    model = build_model(model_name).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    start_time = time.time()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 20)

        for phase in ['training', 'validation']:
            if phase == 'training':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'training'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'training':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'training':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = float(running_corrects.double() / dataset_sizes[phase])

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            if phase == 'training':
                history["train_loss"].append(epoch_loss)
                history["train_acc"].append(epoch_acc)
            else:
                history["val_loss"].append(epoch_loss)
                history["val_acc"].append(epoch_acc)

            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - start_time
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best Val Accuracy: {best_acc:.4f}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_file = MODEL_CHECKPOINTS[model_name]
    save_path = CHECKPOINT_DIR / checkpoint_file
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), save_path)
    
    import json
    hist_file = CHECKPOINT_DIR / "training_history.json"
    all_hist = {}
    if hist_file.exists():
        try:
            with open(hist_file) as f:
                all_hist = json.load(f)
        except Exception:
            all_hist = {}
    all_hist[model_name] = history
    with open(hist_file, "w") as f:
        json.dump(all_hist, f, indent=2)

    print(f"Saved best checkpoint to: {save_path}")
    print(f"Updated training history log in: {hist_file}")

def main():
    parser = argparse.ArgumentParser(description="Train Gallbladder Cancer Detection PyTorch Models")
    parser.add_argument("--models", nargs="+", default=["EfficientNetB0", "ResNet18", "MobileNetV2", "DenseNet121", "GBCNet"],
                        help="Models to train: EfficientNetB0, ResNet18, MobileNetV2, DenseNet121, GBCNet, etc.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs per model")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for dataloaders")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing training/validation/test folders")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_transforms = {
        'training': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'validation': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {
        x: datasets.ImageFolder(os.path.join(args.data_dir, x), data_transforms[x])
        for x in ['training', 'validation', 'test']
    }

    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=(x == 'training'), num_workers=0)
        for x in ['training', 'validation', 'test']
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['training', 'validation', 'test']}
    print(f"Classes: {image_datasets['training'].classes}")
    print(f"Dataset sizes: {dataset_sizes}")

    for model_name in args.models:
        if model_name not in MODEL_CHECKPOINTS:
            print(f"Skipping unknown model: {model_name}")
            continue
        train_single_model(model_name, dataloaders, dataset_sizes, device, num_epochs=args.epochs, lr=args.lr)

if __name__ == "__main__":
    main()
