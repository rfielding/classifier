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
            # Return a dummy image and label
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
    transforms.Normalize([0.485, 0.456, 0.406],   # Mean for ImageNet
                         [0.229, 0.224, 0.225])   # Std for ImageNet
])

# Define the model architecture
class ImageClassifier(pl.LightningModule):
    def __init__(self, num_classes, learning_rate):
        super(ImageClassifier, self).__init__()
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Flatten(),
            nn.Linear(32 * (image_size // 4) * (image_size // 4), 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        # Log training loss and accuracy
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        self.log('train_acc', acc, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        # Log validation loss and accuracy
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        self.log('val_acc', acc, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

def get_label_mappings(img_dir):
    global label_map, idx_to_label
    label_set = set()
    for filename in os.listdir(img_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            label_str = filename.split('.')[0]
            label_set.add(label_str)
    # Sort labels lexically and assign indices
    sorted_labels = sorted(label_set)
    label_map = {label: idx for idx, label in enumerate(sorted_labels)}
    idx_to_label = {idx: label for label, idx in label_map.items()}
    num_classes = len(label_map)
    return num_classes

def prepare_dataset():
    img_paths = []
    img_labels = []
    for filename in os.listdir('./train'):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            filepath = os.path.join('./train', filename)
            label_str = filename.split('.')[0]
            if label_str in label_map:
                label = label_map[label_str]
            else:
                continue  # Skip unknown labels
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
            print(f"Epoch {epoch+1}: val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

def train_model():
    # Get number of classes and label mappings
    num_classes = get_label_mappings('./train')
    if num_classes == 0:
        print("No valid images found in './train' directory for training.")
        sys.exit(1)

    img_paths, img_labels = prepare_dataset()

    # Create dataset
    dataset = CustomImageDataset(img_paths, img_labels, transform=transform)

    # Split into training and validation sets
    total_size = len(dataset)
    val_size = int(validation_split * total_size)
    train_size = total_size - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Set num_workers=11 in DataLoaders
    num_workers = 11

    # Prepare DataLoaders with num_workers
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    # Initialize model
    model = ImageClassifier(num_classes=num_classes, learning_rate=learning_rate)

    # Initialize TensorBoard logger
    logger = TensorBoardLogger("tb_logs", name="classify_model")

    # Initialize checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath='.',
        filename='weights',
        save_top_k=1,
        monitor='val_acc',
        mode='max'
    )

    # Initialize custom callback
    print_callback = PrintValidationMetricsCallback()

    # Initialize trainer with callbacks
    trainer = Trainer(
        max_epochs=num_epochs,
        callbacks=[checkpoint_callback, print_callback],
        logger=logger,
        enable_progress_bar=True,
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    print("Training complete. Model weights saved to 'weights.ckpt'.")

def predict_image(image_path):
    # Get label mappings (needed for idx_to_label)
    num_classes = get_label_mappings('./train')
    if num_classes == 0:
        print("No valid labels found. Please train the model first.")
        sys.exit(1)

    # Initialize model
    model = ImageClassifier(num_classes=num_classes, learning_rate=learning_rate)

    # Load model weights
    checkpoint_path = 'weights.ckpt'
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at '{checkpoint_path}'. Please train the model first using '-train' flag.")
        sys.exit(1)

    model = ImageClassifier.load_from_checkpoint(checkpoint_path, num_classes=num_classes, learning_rate=learning_rate)
    model.eval()

    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension

    # Predict
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)

    # Map numeric prediction to label
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
