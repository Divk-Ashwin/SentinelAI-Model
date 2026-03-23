"""
Train a CNN (MobileNetV2) for phishing image detection using transfer learning.
Loads images from pipeline 2/phishing/ and pipeline 2/legitimate/ directories.
"""

import os
import sys
import glob
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torchvision import models, transforms
from torchvision.models import MobileNet_V2_Weights
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path to import config and utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEVICE, TRAINING_CONFIG, IMAGE_MODEL_CONFIG

# Extract hyperparameters from config
EPOCHS = TRAINING_CONFIG["epochs"]
BATCH_SIZE = TRAINING_CONFIG["batch_size"]
LEARNING_RATE = TRAINING_CONFIG["learning_rate"]
VALIDATION_SPLIT = TRAINING_CONFIG["validation_split"]
RANDOM_SEED = TRAINING_CONFIG["random_seed"]

# Image model configuration
MODEL_ARCH = IMAGE_MODEL_CONFIG["architecture"]
NUM_CLASSES = IMAGE_MODEL_CONFIG["num_classes"]
INPUT_SIZE = IMAGE_MODEL_CONFIG["input_size"]
PRETRAINED = IMAGE_MODEL_CONFIG["pretrained"]

# Set random seeds for reproducibility
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

# Paths (relative to project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE_DIR = os.path.join(PROJECT_ROOT, "pipeline 2")
PHISHING_DIR = os.path.join(PIPELINE_DIR, "phishing")
LEGITIMATE_DIR = os.path.join(PIPELINE_DIR, "legitimate")
SAVE_MODEL_PATH = os.path.join(PROJECT_ROOT, "saved_models/image_model.pth")

# Ensure output directory exists
os.makedirs(os.path.dirname(SAVE_MODEL_PATH), exist_ok=True)

# Image augmentation transforms
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transforms = transforms.Compose([
    transforms.Resize((INPUT_SIZE[0] + 32, INPUT_SIZE[1] + 32)),
    transforms.CenterCrop(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


class PhishingImageDataset(Dataset):
    """PyTorch dataset for phishing image classification."""
    
    def __init__(self, image_paths, labels, transforms=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transforms = transforms
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.warning(f"Failed to load image {img_path}: {e}")
            # Return a blank image instead of crashing
            image = Image.new('RGB', INPUT_SIZE, color=(0, 0, 0))
        
        if self.transforms:
            image = self.transforms(image)
        
        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long)
        }


class MobileNetV2Classifier(nn.Module):
    """MobileNetV2 classifier for binary phishing image classification."""
    
    def __init__(self, num_classes=2, pretrained=True):
        super(MobileNetV2Classifier, self).__init__()
        
        if pretrained:
            weights = MobileNet_V2_Weights.DEFAULT
            self.model = models.mobilenet_v2(weights=weights)
        else:
            self.model = models.mobilenet_v2(weights=None)
        
        # Freeze early layers
        for param in list(self.model.parameters())[:-10]:
            param.requires_grad = False
        
        # Replace classifier
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.model.last_channel, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)


def load_images_from_pipeline():
    """Load image paths and labels from pipeline 2 directories."""

    # Check if directories exist
    if not os.path.exists(PHISHING_DIR) and not os.path.exists(LEGITIMATE_DIR):
        raise FileNotFoundError(
            f"Neither {PHISHING_DIR} nor {LEGITIMATE_DIR} found. "
            f"Please ensure pipeline 2 directory structure exists."
        )

    image_paths = []
    labels = []

    # Load phishing images (label=1)
    if os.path.exists(PHISHING_DIR):
        phishing_images = glob.glob(os.path.join(PHISHING_DIR, "**/*.png"), recursive=True) + \
                         glob.glob(os.path.join(PHISHING_DIR, "**/*.jpg"), recursive=True) + \
                         glob.glob(os.path.join(PHISHING_DIR, "**/*.jpeg"), recursive=True)
        logger.info(f"Found {len(phishing_images)} phishing images")
        image_paths.extend(phishing_images)
        labels.extend([1] * len(phishing_images))
    else:
        logger.warning(f"{PHISHING_DIR} not found")

    # Load legitimate images (label=0)
    if os.path.exists(LEGITIMATE_DIR):
        legitimate_images = glob.glob(os.path.join(LEGITIMATE_DIR, "**/*.png"), recursive=True) + \
                           glob.glob(os.path.join(LEGITIMATE_DIR, "**/*.jpg"), recursive=True) + \
                           glob.glob(os.path.join(LEGITIMATE_DIR, "**/*.jpeg"), recursive=True)
        logger.info(f"Found {len(legitimate_images)} legitimate images")
        image_paths.extend(legitimate_images)
        labels.extend([0] * len(legitimate_images))
    else:
        logger.warning(f"{LEGITIMATE_DIR} not found")

    if not image_paths:
        raise ValueError(
            f"No images found in {PIPELINE_DIR}. "
            f"Expected structure: pipeline 2/phishing/ and pipeline 2/legitimate/"
        )

    logger.info(f"Total images loaded: {len(image_paths)}")
    return np.array(image_paths), np.array(labels)


def train_epoch(model, dataloader, optimizer, criterion):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        
        images = batch["image"].to(DEVICE)
        labels = batch["label"].to(DEVICE)
        
        logits = model(images)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def validate(model, dataloader):
    """Validate model and compute metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            images = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            
            logits = model(images)
            predictions = torch.argmax(logits, dim=1)
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predictions": all_preds,
        "labels": all_labels
    }


def sanity_check_image_model():
    """Verify trained image model can be loaded and makes predictions."""
    logger.info("Running sanity checks...")

    try:
        # Check model file exists
        if not os.path.exists(SAVE_MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {SAVE_MODEL_PATH}")
        logger.info(f"✓ Model file exists: {SAVE_MODEL_PATH}")

        # Load model
        test_model = MobileNetV2Classifier(num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)
        test_model.load_state_dict(torch.load(SAVE_MODEL_PATH, map_location=DEVICE))
        test_model.eval()
        logger.info("✓ Model loaded successfully")

        # Test single sample prediction with a random image
        test_image = Image.new('RGB', INPUT_SIZE, color=(128, 128, 128))
        test_image_tensor = val_transforms(test_image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = test_model(test_image_tensor)
            prediction = torch.argmax(logits, dim=1).item()
            confidence = torch.softmax(logits, dim=1).max().item()

        label_name = "Phishing" if prediction == 1 else "Legitimate"
        logger.info(f"✓ Sample prediction successful (random test image)")
        logger.info(f"  Predicted: {label_name} (confidence: {confidence:.2%})")

        logger.info("✓ All sanity checks passed!")
        return True

    except Exception as e:
        logger.error(f"✗ Sanity check failed: {e}")
        return False


def main():
    logger.info("=" * 70)
    logger.info("Image-based Phishing Detection - CNN (MobileNetV2) Training")
    logger.info("=" * 70)
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Model Architecture: {MODEL_ARCH}")
    logger.info(f"Input Size: {INPUT_SIZE}")
    logger.info(f"Transfer Learning: {PRETRAINED}")
    logger.info(f"Hyperparameters: Epochs={EPOCHS}, Batch Size={BATCH_SIZE}, LR={LEARNING_RATE}")

    # Load images
    logger.info("Loading images from pipeline 2/")
    try:
        image_paths, labels = load_images_from_pipeline()
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Cannot proceed with image training: {e}")
        logger.error("Note: Image training is optional. The API will work without trained image models.")
        logger.error(f"Please ensure pipeline 2/ has the following structure:")
        logger.error(f"  pipeline 2/")
        logger.error(f"    ├── phishing/     (phishing image files)")
        logger.error(f"    └── legitimate/   (legitimate image files)")
        sys.exit(1)

    # Encode labels
    logger.info("Analyzing label distribution...")
    unique, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique, counts):
        label_name = "Phishing" if label == 1 else "Legitimate"
        logger.info(f"  {label_name}: {count} images")

    if len(unique) < 2:
        logger.error("Warning: Only one class found in images. Need both phishing and legitimate images.")
        sys.exit(1)

    # Train/validation split
    logger.info("Creating train/validation split...")
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED, stratify=labels
    )

    logger.info(f"Training set: {len(train_paths)} images")
    logger.info(f"Validation set: {len(val_paths)} images")

    # Create datasets and dataloaders
    logger.info("Creating datasets and dataloaders...")
    train_dataset = PhishingImageDataset(train_paths, train_labels, transforms=train_transforms)
    val_dataset = PhishingImageDataset(val_paths, val_labels, transforms=val_transforms)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Initialize model
    logger.info("Initializing MobileNetV2 model...")
    model = MobileNetV2Classifier(num_classes=NUM_CLASSES, pretrained=PRETRAINED).to(DEVICE)

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Training setup
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    logger.info(f"Training for {EPOCHS} epochs with batch size {BATCH_SIZE}")

    best_val_f1 = 0
    best_epoch = 0

    # Training history
    history = {
        "train_loss": [],
        "val_accuracy": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": []
    }

    for epoch in range(EPOCHS):
        logger.info(f"Epoch {epoch + 1}/{EPOCHS}")

        # Train
        train_loss = train_epoch(model, train_dataloader, optimizer, criterion)
        history["train_loss"].append(train_loss)
        logger.info(f"Training Loss: {train_loss:.4f}")

        # Validate
        val_metrics = validate(model, val_dataloader)
        history["val_accuracy"].append(val_metrics['accuracy'])
        history["val_precision"].append(val_metrics['precision'])
        history["val_recall"].append(val_metrics['recall'])
        history["val_f1"].append(val_metrics['f1'])

        logger.info(f"Validation Accuracy:  {val_metrics['accuracy']:.4f}")
        logger.info(f"Validation Precision: {val_metrics['precision']:.4f}")
        logger.info(f"Validation Recall:    {val_metrics['recall']:.4f}")
        logger.info(f"Validation F1-Score:  {val_metrics['f1']:.4f}")

        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_epoch = epoch + 1
        
        # Save model after every epoch
        logger.info(f"Saving model for epoch {epoch + 1}...")
        epoch_model_path = SAVE_MODEL_PATH.replace('.pth', f'_epoch_{epoch + 1}.pth')
        torch.save(model.state_dict(), epoch_model_path)
        logger.info(f"Model saved to {epoch_model_path}")
        
        # Also save the latest model without epoch number
        torch.save(model.state_dict(), SAVE_MODEL_PATH)
        logger.info(f"Latest model saved to {SAVE_MODEL_PATH}")

    # Final validation metrics
    logger.info("=" * 70)
    logger.info("FINAL EVALUATION RESULTS")
    logger.info("=" * 70)
    logger.info(f"Best Epoch: {best_epoch}/{EPOCHS}")
    logger.info(f"Validation Accuracy:  {val_metrics['accuracy']:.4f}")
    logger.info(f"Validation Precision: {val_metrics['precision']:.4f}")
    logger.info(f"Validation Recall:    {val_metrics['recall']:.4f}")
    logger.info(f"Validation F1-Score:  {val_metrics['f1']:.4f}")

    cm = confusion_matrix(val_metrics['labels'], val_metrics['predictions'])
    logger.info(f"Confusion Matrix - TP: {cm[1, 1]}, TN: {cm[0, 0]}, FP: {cm[0, 1]}, FN: {cm[1, 0]}")

    # Run sanity checks
    if not sanity_check_image_model():
        logger.error("Training completed but sanity checks failed!")
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("Training completed successfully!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
