"""
Train a Feed Forward Neural Network (FFNN) for phishing detection
based on URL and sender metadata features.
Loads CSV files from pipeline 3/ and extracts 15 engineered features.
"""

import os
import sys
import glob
import pickle
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from sklearn.preprocessing import StandardScaler
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
from config import DEVICE, TRAINING_CONFIG, METADATA_MODEL_CONFIG, METADATA_FEATURES
from utils.preprocessing import extract_metadata_features

# Extract hyperparameters from config
EPOCHS = TRAINING_CONFIG["epochs"]
BATCH_SIZE = TRAINING_CONFIG["batch_size"]
LEARNING_RATE = TRAINING_CONFIG["learning_rate"]
VALIDATION_SPLIT = TRAINING_CONFIG["validation_split"]
RANDOM_SEED = TRAINING_CONFIG["random_seed"]

# Set random seeds for reproducibility
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
INPUT_DIM = METADATA_MODEL_CONFIG["input_dim"]
HIDDEN_DIMS = METADATA_MODEL_CONFIG["hidden_dims"]
OUTPUT_DIM = METADATA_MODEL_CONFIG["output_dim"]
DROPOUT = METADATA_MODEL_CONFIG["dropout"]

# Paths (relative to project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE_DIR = os.path.join(PROJECT_ROOT, "pipeline 3")
SAVE_MODEL_PATH = os.path.join(PROJECT_ROOT, "saved_models/metadata_model.pth")
SAVE_SCALER_PATH = os.path.join(PROJECT_ROOT, "saved_models/metadata_scaler.pkl")

# Ensure output directory exists
os.makedirs(os.path.dirname(SAVE_MODEL_PATH), exist_ok=True)


class MetadataDataset(Dataset):
    """PyTorch dataset for metadata-based phishing detection."""
    
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return {
            "features": self.features[idx],
            "labels": self.labels[idx]
        }


class MetadataFFNN(nn.Module):
    """Feed Forward Neural Network for metadata-based phishing detection."""
    
    def __init__(self, input_dim, hidden_dims, output_dim, dropout):
        super(MetadataFFNN, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def load_data_from_pipeline():
    """Load and combine CSV files from pipeline 3 directory."""
    csv_files = glob.glob(os.path.join(PIPELINE_DIR, "*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {PIPELINE_DIR}")
    
    logger.info(f"Found {len(csv_files)} CSV files")

    dataframes = []
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        logger.info(f"Loading {filename}...")
        
        # Try different encodings
        df = None
        for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
            try:
                df = pd.read_csv(csv_file, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.error(f"Error reading {filename} with {encoding}: {e}")
                break
        
        if df is None:
            logger.error(f"Could not read {filename} with any encoding. Skipping.")
            continue
        
        try:
            # Normalize column names to lowercase
            df.columns = df.columns.str.lower()
            
            # Handle different dataset formats
            if "huggingface" in filename.lower():
                # HuggingFace: has 'label' column
                if "label" in df.columns:
                    dataframes.append(df)
                else:
                    logger.warning(f"{filename} doesn't have 'label' column. Skipping.")
            
            elif "iscx" in filename.lower():
                # ISCX: last column is the label (URL_Type_obf_Type)
                last_col = df.columns[-1]
                df = df.rename(columns={last_col: "label"})
                dataframes.append(df)
            
            elif "phiusiil" in filename.lower():
                # PhiUSIIL: has 'label' column
                if "label" in df.columns:
                    dataframes.append(df)
                else:
                    logger.warning(f"{filename} doesn't have 'label' column. Skipping.")
            
            else:
                # Generic fallback: check for label column
                if "label" in df.columns:
                    dataframes.append(df)
                else:
                    logger.warning(f"{filename} doesn't have expected columns. Skipping.")
                    
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")

    if not dataframes:
        raise ValueError("No valid CSV files could be loaded")

    combined_df = pd.concat(dataframes, ignore_index=True)
    logger.info(f"Total records loaded: {len(combined_df)}")
    
    return combined_df


def encode_labels(df):
    """Encode labels as 0 (legitimate) and 1 (phishing)."""
    # Normalize label values
    if "label" in df.columns:
        df["label"] = df["label"].astype(str).str.lower().str.strip()
    else:
        raise ValueError("No 'label' column found in data")

    label_mapping = {
        "phishing": 1, "legitimate": 0, "non-phishing": 0,
        "ham": 0, "spam": 1, "1": 1, "0": 0, "yes": 1, "no": 0
    }

    df["label"] = df["label"].map(lambda x: label_mapping.get(str(x).lower(), -1))

    # Remove any rows with unmapped labels
    df = df[df["label"] != -1]

    logger.info(f"Label distribution:\n{df['label'].value_counts()}")
    
    return df


def extract_features_batch(df):
    """Extract or use existing metadata features."""
    # Remove label column and NaN values
    feature_cols = [col for col in df.columns if col != 'label']
    features_df = df[feature_cols].copy()
    
    # Check if we have URL/sender/timestamp columns (raw data) or pre-computed features
    has_raw_data = any(col in features_df.columns for col in ['url', 'sender', 'timestamp'])
    
    if has_raw_data:
        # If raw data exists, extract features using preprocessing function
        logger.info("Extracting features from raw data...")
        features_list = []
        
        for idx, row in tqdm(features_df.iterrows(), total=len(features_df), desc="Extracting features"):
            url = row.get("url", None) if pd.notna(row.get("url")) else None
            sender = row.get("sender", None) if pd.notna(row.get("sender")) else None
            timestamp = row.get("timestamp", None) if pd.notna(row.get("timestamp")) else None
            
            features = extract_metadata_features(url=url, sender=sender, timestamp=timestamp)
            features_list.append(features)
        
        features_df = pd.DataFrame(features_list)
        
        # Ensure all 15 features are present and in correct order
        features_df = features_df[METADATA_FEATURES]
    else:
        # Pre-computed features: select numeric columns and drop non-numeric ones
        logger.info("Using pre-computed features from dataset...")
        
        # Convert all columns to numeric, coercing errors
        for col in features_df.columns:
            features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
        
        # Drop rows with NaN values
        features_df = features_df.dropna()
        
        logger.info(f"Using {len(features_df.columns)} features from dataset")
    
    return features_df.values


def train_epoch(model, dataloader, optimizer, criterion):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        
        features = batch["features"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        
        logits = model(features)
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
            features = batch["features"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            
            logits = model(features)
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


def sanity_check_metadata_model():
    """Verify trained metadata model can be loaded and makes predictions."""
    logger.info("Running sanity checks...")

    try:
        # Check model file exists
        if not os.path.exists(SAVE_MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {SAVE_MODEL_PATH}")
        logger.info(f"✓ Model file exists: {SAVE_MODEL_PATH}")

        # Check scaler file exists
        if not os.path.exists(SAVE_SCALER_PATH):
            raise FileNotFoundError(f"Scaler file not found at {SAVE_SCALER_PATH}")
        logger.info(f"✓ Scaler file exists: {SAVE_SCALER_PATH}")

        # Load scaler
        with open(SAVE_SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        logger.info("✓ Scaler loaded successfully")

        # Load model
        test_model = MetadataFFNN(INPUT_DIM, HIDDEN_DIMS, OUTPUT_DIM, DROPOUT).to(DEVICE)
        test_model.load_state_dict(torch.load(SAVE_MODEL_PATH, map_location=DEVICE))
        test_model.eval()
        logger.info("✓ Model loaded successfully")

        # Test single sample prediction
        test_features = np.random.randn(1, INPUT_DIM).astype(np.float32)
        test_features_normalized = scaler.transform(test_features)

        with torch.no_grad():
            features_tensor = torch.tensor(test_features_normalized, dtype=torch.float32).to(DEVICE)
            logits = test_model(features_tensor)
            prediction = torch.argmax(logits, dim=1).item()
            confidence = torch.softmax(logits, dim=1).max().item()

        label_name = "Phishing" if prediction == 1 else "Legitimate"
        logger.info(f"✓ Sample prediction successful (random test features)")
        logger.info(f"  Predicted: {label_name} (confidence: {confidence:.2%})")

        logger.info("✓ All sanity checks passed!")
        return True

    except Exception as e:
        logger.error(f"✗ Sanity check failed: {e}")
        return False


def main():
    logger.info("=" * 70)
    logger.info("Metadata-based Phishing Detection - FFNN Training")
    logger.info("=" * 70)
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Model Architecture: {INPUT_DIM} -> {' -> '.join(map(str, HIDDEN_DIMS))} -> {OUTPUT_DIM}")
    logger.info(f"Hyperparameters: Epochs={EPOCHS}, Batch Size={BATCH_SIZE}, LR={LEARNING_RATE}")
    logger.info(f"Features: {len(METADATA_FEATURES)} metadata features")

    # Load data
    logger.info("Loading data from pipeline 3/")
    df = load_data_from_pipeline()

    # Encode labels
    logger.info("Encoding labels...")
    df = encode_labels(df)

    # Remove duplicates and null values
    df = df.dropna(subset=["label"])
    logger.info(f"Data after cleaning: {len(df)} records")

    # Extract metadata features
    logger.info("Extracting metadata features...")
    features = extract_features_batch(df)
    labels = df["label"].values

    logger.info(f"Features shape: {features.shape}")
    actual_input_dim = features.shape[1]
    
    # Update INPUT_DIM if it doesn't match
    if actual_input_dim != INPUT_DIM:
        logger.warning(f"Feature dimension mismatch: config={INPUT_DIM}, actual={actual_input_dim}")
        logger.info(f"Using actual feature dimension: {actual_input_dim}")
        ACTUAL_INPUT_DIM = actual_input_dim
    else:
        ACTUAL_INPUT_DIM = INPUT_DIM

    # Normalize features
    logger.info("Normalizing features using StandardScaler...")
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    logger.info(f"Scaler fitted on {features.shape[0]} samples")

    # Train/validation split
    logger.info("Creating train/validation split...")
    from sklearn.model_selection import train_test_split

    train_features, val_features, train_labels, val_labels = train_test_split(
        features, labels, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED, stratify=labels
    )

    logger.info(f"Training set: {len(train_features)} samples")
    logger.info(f"Validation set: {len(val_features)} samples")

    # Create datasets and dataloaders
    logger.info("Creating datasets and dataloaders...")
    train_dataset = MetadataDataset(train_features, train_labels)
    val_dataset = MetadataDataset(val_features, val_labels)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model with actual input dimension
    logger.info("Initializing FFNN model...")
    model = MetadataFFNN(ACTUAL_INPUT_DIM, HIDDEN_DIMS, OUTPUT_DIM, DROPOUT).to(DEVICE)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")

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
        logger.info(f"Epoch {epoch + 1} - Training Loss: {train_loss:.4f}")
        
        # Validate
        val_metrics = validate(model, val_dataloader)
        history["val_accuracy"].append(val_metrics['accuracy'])
        history["val_precision"].append(val_metrics['precision'])
        history["val_recall"].append(val_metrics['recall'])
        history["val_f1"].append(val_metrics['f1'])
        
        logger.info(f"Epoch {epoch + 1} - Val Accuracy: {val_metrics['accuracy']:.4f}, "
                   f"Precision: {val_metrics['precision']:.4f}, "
                   f"Recall: {val_metrics['recall']:.4f}, "
                   f"F1: {val_metrics['f1']:.4f}")
        
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
        
        # Save scaler (only once, same for all epochs)
        if epoch == 0:
            with open(SAVE_SCALER_PATH, 'wb') as f:
                pickle.dump(scaler, f)
            logger.info(f"Scaler saved to {SAVE_SCALER_PATH}")
    
    # Final validation metrics
    logger.info("FINAL EVALUATION RESULTS")
    logger.info(f"Best Epoch: {best_epoch}/{EPOCHS}")
    logger.info(f"Validation Metrics - Accuracy: {val_metrics['accuracy']:.4f}, "
               f"Precision: {val_metrics['precision']:.4f}, "
               f"Recall: {val_metrics['recall']:.4f}, "
               f"F1-Score: {val_metrics['f1']:.4f}")
    
    cm = confusion_matrix(val_metrics['labels'], val_metrics['predictions'])
    logger.info(f"Confusion Matrix - TN: {cm[0, 0]}, FP: {cm[0, 1]}, FN: {cm[1, 0]}, TP: {cm[1, 1]}")
    
    # Training summary
    logger.info(f"Training Summary - Final Loss: {history['train_loss'][-1]:.4f}, "
               f"Min Loss: {min(history['train_loss']):.4f} (Epoch {history['train_loss'].index(min(history['train_loss'])) + 1}), "
               f"Max F1: {max(history['val_f1']):.4f} (Epoch {history['val_f1'].index(max(history['val_f1'])) + 1})")
    
    logger.info(f"Final Results - Accuracy: {val_metrics['accuracy']:.4f}, "
               f"Precision: {val_metrics['precision']:.4f}, "
               f"Recall: {val_metrics['recall']:.4f}, "
               f"F1: {val_metrics['f1']:.4f}")
    
    # Run sanity checks
    if not sanity_check_metadata_model():
        logger.error("Training completed but sanity checks failed!")
        sys.exit(1)
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
