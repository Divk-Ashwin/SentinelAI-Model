"""
Fine-tune a multilingual BERT model (XLM-RoBERTa) for SMS spam detection.
Loads multiple CSV files from pipeline 1/ and trains on spam/ham classification.
"""

import os
import sys
import glob
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path to import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEVICE, TRAINING_CONFIG, TEXT_MODEL_CONFIG

# Extract hyperparameters from config
EPOCHS = TRAINING_CONFIG["epochs"]
BATCH_SIZE = TRAINING_CONFIG["batch_size"]
LEARNING_RATE = TRAINING_CONFIG["learning_rate"]
MAX_SEQ_LENGTH = TRAINING_CONFIG["max_sequence_length"]
VALIDATION_SPLIT = TRAINING_CONFIG["validation_split"]
RANDOM_SEED = TRAINING_CONFIG["random_seed"]

# Model configuration
MODEL_NAME = TEXT_MODEL_CONFIG["model_name"]
DROPOUT = TEXT_MODEL_CONFIG["dropout"]

# Paths (relative to project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE_DIR = os.path.join(PROJECT_ROOT, "pipeline 1")
SAVE_MODEL_PATH = os.path.join(PROJECT_ROOT, "saved_models/text_model.pth")
SAVE_TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "saved_models/text_tokenizer")


class SpamDataset(Dataset):
    """PyTorch dataset for spam detection."""
    
    def __init__(self, texts, labels, tokenizer, max_length=MAX_SEQ_LENGTH):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long)
        }


class SpamClassifier(nn.Module):
    """Binary classifier on top of BERT."""
    
    def __init__(self, model_name, dropout=DROPOUT):
        super(SpamClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use CLS token representation (always available)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


def load_data_from_pipeline():
    """Load and combine CSV files from pipeline 1 directory."""
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
            if "kaggle" in filename.lower():
                # Kaggle: labels, text
                if "labels" in df.columns and "text" in df.columns:
                    dataframes.append(df[["labels", "text"]].rename(columns={"labels": "label"}))
                else:
                    logger.warning(f"{filename} doesn't have expected columns. Skipping.")
            
            elif "mendeley" in filename.lower():
                # Mendeley: LABEL, TEXT
                if "label" in df.columns and "text" in df.columns:
                    dataframes.append(df[["label", "text"]])
                else:
                    logger.warning(f"{filename} doesn't have expected columns. Skipping.")
            
            elif "smishtank" in filename.lower():
                # smishtank: uses Phishing label and MainText
                if "phishing" in df.columns and "maintext" in df.columns:
                    # Convert phishing (boolean/int) to binary: 1=spam, 0=ham
                    df_sub = df[["phishing", "maintext"]].copy()
                    # Drop rows with NaN values
                    df_sub = df_sub.dropna()
                    df_sub.columns = ["label", "text"]
                    df_sub["label"] = df_sub["label"].astype(int)
                    dataframes.append(df_sub)
                else:
                    logger.warning(f"{filename} doesn't have expected columns. Skipping.")
            
            elif "uci" in filename.lower():
                # UCI: v1 is label, v2 is text
                if "v1" in df.columns and "v2" in df.columns:
                    df_sub = df[["v1", "v2"]].copy()
                    df_sub.columns = ["label", "text"]
                    dataframes.append(df_sub)
                else:
                    logger.warning(f"{filename} doesn't have expected columns. Skipping.")
            
            else:
                # Generic fallback: look for label and text
                if "label" in df.columns and "text" in df.columns:
                    dataframes.append(df[["label", "text"]])
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
    """Encode labels as 0 (ham) and 1 (spam)."""
    # Normalize label values
    df["label"] = df["label"].str.lower().str.strip()
    label_mapping = {
        "spam": 1, "ham": 0, "phishing": 1,
        "legitimate": 0, "1": 1, "0": 0,
        "non-phishing": 0, "non-spam": 0
    }

    df["label"] = df["label"].map(lambda x: label_mapping.get(str(x).lower(), -1))

    # Remove any rows with unmapped labels
    df = df[df["label"] != -1]

    logger.info(f"Label distribution:\n{df['label'].value_counts()}")

    return df


def train_epoch(model, dataloader, optimizer, criterion):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def validate(model, dataloader):
    """Validate model on a dataset."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            
            logits = model(input_ids, attention_mask)
            predictions = torch.argmax(logits, dim=1)
            
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total if total > 0 else 0
    return accuracy


def sanity_check_text_model():
    """Verify trained text model can be loaded and makes predictions."""
    logger.info("Running sanity checks...")

    try:
        # Check model file exists
        if not os.path.exists(SAVE_MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {SAVE_MODEL_PATH}")
        logger.info(f"✓ Model file exists: {SAVE_MODEL_PATH}")

        # Check tokenizer directory exists
        if not os.path.exists(SAVE_TOKENIZER_PATH):
            raise FileNotFoundError(f"Tokenizer directory not found at {SAVE_TOKENIZER_PATH}")
        logger.info(f"✓ Tokenizer directory exists: {SAVE_TOKENIZER_PATH}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(SAVE_TOKENIZER_PATH)
        logger.info("✓ Tokenizer loaded successfully")

        # Load model
        test_model = SpamClassifier(MODEL_NAME).to(DEVICE)
        test_model.load_state_dict(torch.load(SAVE_MODEL_PATH, map_location=DEVICE))
        test_model.eval()
        logger.info("✓ Model loaded successfully")

        # Test single sample prediction
        test_text = "Click here to verify your account"
        encoding = tokenizer(
            test_text,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        with torch.no_grad():
            input_ids = encoding["input_ids"].to(DEVICE)
            attention_mask = encoding["attention_mask"].to(DEVICE)
            logits = test_model(input_ids, attention_mask)
            prediction = torch.argmax(logits, dim=1).item()
            confidence = torch.softmax(logits, dim=1).max().item()

        label_name = "Spam" if prediction == 1 else "Ham"
        logger.info(f"✓ Sample prediction successful: '{test_text}'")
        logger.info(f"  Predicted: {label_name} (confidence: {confidence:.2%})")

        logger.info("✓ All sanity checks passed!")
        return True

    except Exception as e:
        logger.error(f"✗ Sanity check failed: {e}")
        return False


def main():
    logger.info("=" * 60)
    logger.info("SMS Spam Detection - Text Model Training")
    logger.info("=" * 60)
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Hyperparameters: Epochs={EPOCHS}, Batch Size={BATCH_SIZE}, LR={LEARNING_RATE}")
    logger.info(f"Max Sequence Length: {MAX_SEQ_LENGTH}")

    # Load data
    logger.info("Loading data from pipeline 1/")
    df = load_data_from_pipeline()

    # Encode labels
    logger.info("Encoding labels...")
    df = encode_labels(df)

    # Remove duplicates and null values
    df = df.dropna(subset=["label", "text"])
    df = df.drop_duplicates(subset=["text"])
    logger.info(f"Data after cleaning: {len(df)} records")

    # Train/validation split
    logger.info("Creating train/validation split...")
    texts = df["text"].values
    labels = df["label"].values

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED, stratify=labels
    )

    logger.info(f"Training set: {len(train_texts)} samples")
    logger.info(f"Validation set: {len(val_texts)} samples")

    # Load tokenizer and create datasets
    logger.info("Loading tokenizer and creating datasets...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = SpamDataset(train_texts, train_labels, tokenizer)
    val_dataset = SpamDataset(val_texts, val_labels, tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model
    logger.info("Initializing model...")
    model = SpamClassifier(MODEL_NAME).to(DEVICE)
    logger.info(f"Model moved to device: {DEVICE}")

    # Training setup
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    logger.info(f"Training for {EPOCHS} epochs with batch size {BATCH_SIZE}")

    for epoch in range(EPOCHS):
        logger.info(f"Epoch {epoch + 1}/{EPOCHS}")

        # Train
        train_loss = train_epoch(model, train_dataloader, optimizer, criterion)
        logger.info(f"Training Loss: {train_loss:.4f}")

        # Validate
        val_accuracy = validate(model, val_dataloader)
        logger.info(f"Validation Accuracy: {val_accuracy:.4f}")

    # Save model
    logger.info("Saving model and tokenizer...")
    os.makedirs(os.path.dirname(SAVE_MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), SAVE_MODEL_PATH)
    logger.info(f"Model saved to {SAVE_MODEL_PATH}")

    # Save tokenizer
    os.makedirs(SAVE_TOKENIZER_PATH, exist_ok=True)
    tokenizer.save_pretrained(SAVE_TOKENIZER_PATH)
    logger.info(f"Tokenizer saved to {SAVE_TOKENIZER_PATH}")

    # Run sanity checks
    if not sanity_check_text_model():
        logger.error("Training completed but sanity checks failed!")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Training completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
