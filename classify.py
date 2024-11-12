import os
import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from PIL import Image
from torch.utils.data import random_split, DataLoader
import argparse
import json

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

batch_size = config.get('batch_size', 64)
learning_rate = config.get('learning_rate', 0.001)
num_epochs = config.get('num_epochs', 10)
image_size = config.get('image_size', 224)
validation_split = config.get('validation_split', 0.2)

# Global variable to store label mapping
label_map = {}
idx_to_label = {}

# Define custom dataset
class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, img_paths, img_labels, transform=None):
        self.img_labels = img_labels
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.img_paths[idx]).convert('RGB')
        except Exception as e:
            print(f"Error loading image {self.img_paths[idx]}: {e}")
            image = Image.new('RGB', (image_size, image_size))
            label = 0
        else:
            if self.transform:
                image = self.transform(image)
            label = self.img_labels[idx]
        return image, label

# Data transformations
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Mean and std for ImageNet
])

class DeeperImageClassifier(pl.LightningModule):
    def __init__(self, num_classes, learning_rate):
        super(DeeperImageClassifier, self).__init__()
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(512 * (image_size // 64) * (image_size // 64), 1024),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(256, num_classes)
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        self.log('train_acc', acc, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        self.log('val_acc', acc, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

def get_label_mappings(img_dir):
    global label_map, idx_to_label
    label_set = set()
    for filename in os.listdir(img_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            label_str = filename.split('.')[0]
            label_set.add(label_str)
    sorted_labels = sorted(label_set)
    label_map = {label: idx for idx, label in enumerate(sorted_labels)}
    idx_to_label = {idx: label for label, idx in label_map.items()}
    return len(label_map)

def prepare_dataset():
    img_paths = []
    img_labels = []
    for filename in os.listdir('./train'):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            filepath = os.path.join('./train', filename)
            label_str = filename.split('.')[0]
            if label_str in label_map:
                label = label_map[label_str]
                img_paths.append(filepath)
                img_labels.append(label)
    return img_paths, img_labels

# Custom callback to print validation metrics
class PrintValidationMetricsCallback(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get('val_loss')
        val_acc = trainer.callback_metrics.get('val_acc')
        epoch = trainer.current_epoch
        if val_loss is not None and val_acc is not None:
            print(f"Epoch {epoch+1}: val_acc={val_acc:.4f}")

def train_model():
    num_classes = get_label_mappings('./train')
    img_paths, img_labels = prepare_dataset()
    dataset = CustomImageDataset(img_paths, img_labels, transform=transform)
    total_size = len(dataset)
    val_size = int(validation_split * total_size)
    train_size = total_size - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    num_workers = 11

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = DeeperImageClassifier(num_classes=num_classes, learning_rate=learning_rate)
    logger = TensorBoardLogger("tb_logs", name="classify_model")

    print_callback = PrintValidationMetricsCallback()
    trainer = Trainer(
        max_epochs=num_epochs,
        callbacks=[print_callback],
        logger=logger,
        enable_progress_bar=True,
    )

    trainer.fit(model, train_loader, val_loader)

    torch.save(model, "full_model.pth")
    print("Training complete. Entire model saved to 'full_model.pth'.")

def predict_image(image_path):
    num_classes = get_label_mappings('./train')
    model_path = "full_model.pth"
    if not os.path.exists(model_path):
        print(f"No model file found at '{model_path}'. Please train the model first using '-train' flag.")
        sys.exit(1)

    model = torch.load(model_path)
    model.eval()

    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)

    predicted_label = idx_to_label.get(predicted.item(), 'Unknown')
    print(f"Predicted label: {predicted_label}")

def main():
    parser = argparse.ArgumentParser(description='Image Classification Utility')
    parser.add_argument('-train', action='store_true', help='Train the model')
    parser.add_argument('image', nargs='?', help='Image file to classify')

    args = parser.parse_args()

    if args.train:
        train_model()
    elif args.image:
        image_path = args.image
        if not os.path.exists(image_path):
            print(f"Image file '{image_path}' does not exist.")
            sys.exit(1)
        predict_image(image_path)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()

