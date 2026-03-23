"""
Fine-tune a multilingual BERT model (XLM-RoBERTa) for SMS spam detection.
Loads multiple CSV files from pipeline 1/ and trains on spam/ham classification.
"""

import os
import sys
import glob
import shutil
import logging
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import resample
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
WARMUP_STEPS = TRAINING_CONFIG["warmup_steps"]
EARLY_STOPPING_PATIENCE = TRAINING_CONFIG["early_stopping_patience"]

# Model configuration
MODEL_NAME = TEXT_MODEL_CONFIG["model_name"]
DROPOUT = TEXT_MODEL_CONFIG["dropout"]

# Paths (relative to project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE_DIR = os.path.join(PROJECT_ROOT, "pipeline 1")
SAVE_MODEL_PATH = os.path.join(PROJECT_ROOT, "saved_models/text_model.pth")
SAVE_MODEL_BEST_PATH = os.path.join(PROJECT_ROOT, "saved_models/text_model_best.pth")
SAVE_MODEL_LAST_PATH = os.path.join(PROJECT_ROOT, "saved_models/text_model_last.pth")
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


def generate_synthetic_hard_negatives(num_samples: int = 200) -> pd.DataFrame:
    """
    Generate synthetic legitimate messages (hard negatives) to reduce false positives.
    Creates variations of OTP, bank alerts, delivery notifications, and UPI confirmations.

    Args:
        num_samples: Total number of synthetic samples to generate (default 200)

    Returns:
        DataFrame with 'text' and 'label' (label=0 for legitimate)
    """
    logger.info(f"Generating {num_samples} synthetic hard negative samples...")

    # Set seed for reproducibility
    random.seed(42)

    banks = ["HDFC Bank", "SBI", "ICICI Bank", "Axis Bank", "Kotak Bank", "PNB", "Bank of Baroda"]
    stores = ["Amazon", "Flipkart", "Myntra", "Snapdeal", "Paytm Mall"]
    names = ["Rahul", "Priya", "Amit", "Neha", "Vikram", "Sneha", "Arjun", "Pooja"]

    templates = {
        "otp": [
            "Your OTP for {bank} is {otp}. Valid for 10 minutes. Do not share this OTP with anyone.",
            "{otp} is your One Time Password (OTP) for {bank} transaction. Valid for {mins} mins. Do not share.",
            "Dear Customer, your OTP for transaction verification is {otp}. Valid for {mins} minutes. - {bank}",
            "Use {otp} as your OTP for {bank} transaction. This OTP is valid for next {mins} minutes.",
        ],
        "bank_alert": [
            "INR {amount} debited from your {bank} account XX{digits}. Available balance: INR {balance}.",
            "Your {bank} a/c XX{digits} is debited with Rs.{amount} on {date}. Avbl bal: Rs.{balance}.",
            "Rs {amount} has been debited from your {bank} account ending {digits}. Current balance Rs {balance}.",
            "Alert: Rs.{amount} debited from {bank} account XX{digits}. Available bal: Rs.{balance}.",
        ],
        "delivery": [
            "Your {store} order #{order_id} has been delivered. Rate your experience at {store}.in/feedback",
            "Your package from {store} has been delivered to your address. Order #{order_id}",
            "{store} order #{order_id} delivered successfully. Track at {store}.in/orders",
            "Delivered: Your {store} order #{order_id}. Thank you for shopping with us!",
        ],
        "upi": [
            "Payment of Rs.{amount} received from {name} via UPI. Ref: {ref}.",
            "You have received Rs {amount} from {name} to your UPI ID. Txn ID: {ref}",
            "Rs.{amount} credited to your account from {name} via UPI. Reference number: {ref}",
            "UPI payment received: Rs {amount} from {name}. Transaction ref: {ref}",
        ]
    }

    synthetic_texts = []
    samples_per_type = num_samples // 4

    # Generate OTP messages
    for _ in range(samples_per_type):
        template = random.choice(templates["otp"])
        text = template.format(
            bank=random.choice(banks),
            otp=random.randint(100000, 999999),
            mins=random.choice([5, 10, 15])
        )
        synthetic_texts.append(text)

    # Generate bank alerts
    for _ in range(samples_per_type):
        template = random.choice(templates["bank_alert"])
        text = template.format(
            bank=random.choice(banks),
            amount=random.randint(100, 50000),
            balance=random.randint(1000, 100000),
            digits=random.randint(1000, 9999),
            date=f"{random.randint(1,28)}-{random.randint(1,12)}-2026"
        )
        synthetic_texts.append(text)

    # Generate delivery notifications
    for _ in range(samples_per_type):
        template = random.choice(templates["delivery"])
        order_id = f"{random.randint(100,999)}-{random.randint(1000000,9999999)}-{random.randint(1000000,9999999)}"
        text = template.format(
            store=random.choice(stores),
            order_id=order_id
        )
        synthetic_texts.append(text)

    # Generate UPI confirmations
    for _ in range(samples_per_type):
        template = random.choice(templates["upi"])
        text = template.format(
            amount=random.randint(10, 10000),
            name=random.choice(names),
            ref=random.randint(100000000000, 999999999999)
        )
        synthetic_texts.append(text)

    # Create DataFrame
    synthetic_df = pd.DataFrame({
        'text': synthetic_texts,
        'label': 0  # All synthetic samples are legitimate (HAM)
    })

    logger.info(f"✓ Generated {len(synthetic_df)} synthetic legitimate messages")
    return synthetic_df


def load_and_prepare_data():
    """
    Load data from pipeline and clean it with minimal row loss.

    Label normalization:
    - Label 1 (spam/malicious): "spam", "phishing", "smishing", "1", 1
    - Label 0 (ham/safe): "ham", "legitimate", "safe", "0", 0, "non-phishing", "non-spam"

    Only drops rows where:
    - Label is null or unrecognizable
    - Text is completely empty or null
    """
    # Step 1: Load raw data
    df = load_data_from_pipeline()
    total_raw = len(df)
    logger.info(f"[Step 1] Raw data loaded: {total_raw} rows")

    # Step 1.5: Load translated multilingual data (Hindi/Telugu) if available
    translated_csv = os.path.join(PIPELINE_DIR, "translated_phishing.csv")
    if os.path.exists(translated_csv):
        try:
            translated_df = pd.read_csv(translated_csv, encoding='utf-8')
            translated_df.columns = translated_df.columns.str.lower()
            if "label" in translated_df.columns and "text" in translated_df.columns:
                df = pd.concat([df, translated_df], ignore_index=True)
                logger.info(f"[Step 1.5] Loaded translated data: {len(translated_df)} rows (Hindi/Telugu)")
        except Exception as e:
            logger.warning(f"Could not load translated data: {e}")

    # Step 1.6: Add hardcoded global phishing/ham samples for better coverage
    logger.info("[Step 1.6] Adding hardcoded global phishing and ham samples...")
    global_phishing = [
        "Your PayPal account has been limited. Verify: paypal-secure.xyz",
        "IRS: Tax refund $2,400 pending. Confirm: irs-refund.net",
        "Netflix: Payment failed. Update billing: netflix-update.com",
        "DHL: Package held. Pay customs: dhl-delivery.xyz/pay",
        "Amazon: Unusual signin. Verify: amazon-security.net",
        "WhatsApp account expires today. Renew: whatsapp-renew.com",
        "HSBC: Suspicious transaction. Call now: 0800-fake-number",
        "You won iPhone 15. Claim: apple-winner.xyz/claim",
        "Google account accessed from Russia. Secure: google-alert.net",
        "FedEx delivery failed. Reschedule: fedex-redeliver.com",
        "Your SBI account blocked. Verify KYC: sbi-kyc-update.xyz",
        "URGENT: UPI suspended. Update now: paytm-kyc.xyz",
        "Jio Lucky Draw: Won Rs.50000. Claim now: jio-prize.net",
        "HDFC card blocked. Verify: hdfc-secure-verify.xyz",
        "ICICI Account suspended. Restore: icici-restore.net",
        "Trump Stimulus Check approved. Confirm details: stimulus-check.xyz",
        "Your account will be deactivated. Click: secure-verify.net",
        "FREE: You have been selected for cash prize. Claim now",
        "TRAI: Your mobile will be disconnected. Verify: trai-verify.xyz",
        "Aadhaar KYC pending. Update now or service stops"
    ]

    global_ham = [
        "Your Amazon order has been delivered successfully.",
        "PayPal: You sent $50.00 to John. Transaction ID: ABC123",
        "Netflix: Your payment of $15.99 was successful",
        "DHL: Package delivered. Thank you for using DHL",
        "Google: New sign-in from Chrome on Windows. Was this you?",
        "Your Jio recharge of Rs.299 is successful. Validity: 28 days",
        "HDFC Bank: Rs.1000 credited to your account XX1234",
        "Flight PNR ABC123 confirmed. Check-in opens 24hrs before",
        "Your Swiggy order is on the way. ETA 15 minutes",
        "Zomato: Order delivered. Rate your experience",
        "Airtel: Your bill of Rs.499 is due on March 30",
        "LIC Premium of Rs.5000 deducted successfully"
    ]

    hardcoded_data = []
    for text in global_phishing:
        hardcoded_data.append({"text": text, "label": 1})
    for text in global_ham:
        hardcoded_data.append({"text": text, "label": 0})

    hardcoded_df = pd.DataFrame(hardcoded_data)
    df = pd.concat([df, hardcoded_df], ignore_index=True)
    logger.info(f"[Step 1.6] Added {len(hardcoded_data)} hardcoded samples ({len(global_phishing)} spam, {len(global_ham)} ham)")

    # Step 2: Drop rows with null labels
    before_null_label = len(df)
    df = df[df["label"].notna()]
    after_null_label = len(df)
    logger.info(f"[Step 2] After dropping null labels: {after_null_label} rows (dropped {before_null_label - after_null_label})")

    # Step 3: Normalize labels - handle both string and numeric types
    # Define mapping for spam (1) and ham (0)
    spam_variants = {"spam", "phishing", "smishing", "1"}
    ham_variants = {"ham", "legitimate", "safe", "0", "non-phishing", "non-spam"}

    def normalize_label(label):
        """Convert label to 0 (ham) or 1 (spam), or -1 if unrecognized."""
        # Handle numeric labels directly
        if isinstance(label, (int, float)):
            if label == 1 or label == 1.0:
                return 1
            elif label == 0 or label == 0.0:
                return 0
            # Multi-class phishing labels (>=2) map to spam (1)
            elif label >= 2:
                return 1
            else:
                return -1

        # Handle string labels
        label_str = str(label).lower().strip()
        if label_str in spam_variants:
            return 1
        elif label_str in ham_variants:
            return 0
        # Check if string is a numeric >= 2 (for Mendeley multi-class)
        try:
            num_val = float(label_str)
            if num_val >= 2:
                return 1
        except ValueError:
            pass
        return -1

    df["label"] = df["label"].apply(normalize_label)

    # Step 4: Drop rows with unrecognized labels
    before_unrecognized = len(df)
    unrecognized_mask = df["label"] == -1
    unrecognized_count = unrecognized_mask.sum()
    if unrecognized_count > 0:
        # Log sample of unrecognized labels for debugging
        sample_unrecognized = df.loc[unrecognized_mask, "label"].head(5).tolist()
        logger.warning(f"Found {unrecognized_count} rows with unrecognized labels. Samples: {sample_unrecognized}")

    df = df[df["label"] != -1]
    after_unrecognized = len(df)
    logger.info(f"[Step 3-4] After normalizing & dropping unrecognized labels: {after_unrecognized} rows (dropped {before_unrecognized - after_unrecognized})")

    # Step 5: Drop rows with null or empty text only
    before_null_text = len(df)
    # Keep rows where text is not null AND not empty/whitespace-only
    df["text"] = df["text"].astype(str)  # Convert to string first
    df = df[df["text"].notna() & (df["text"].str.strip() != "") & (df["text"] != "nan")]
    after_null_text = len(df)
    logger.info(f"[Step 5] After dropping null/empty text: {after_null_text} rows (dropped {before_null_text - after_null_text})")

    # Step 6: Smart deduplication - prioritize spam label when same text has different labels
    before_dedup = len(df)
    # Group by text and take max label (1=spam takes priority over 0=ham)
    df = df.groupby("text", as_index=False).agg({"label": "max"})
    after_dedup = len(df)
    logger.info(f"[Step 6] After smart deduplication (spam priority): {after_dedup} rows (dropped {before_dedup - after_dedup})")

    # Step 7: Add synthetic hard negatives (legitimate messages)
    logger.info("[Step 7] Adding synthetic hard negative samples...")
    synthetic_df = generate_synthetic_hard_negatives(num_samples=200)
    before_synthetic = len(df)
    df = pd.concat([df, synthetic_df], ignore_index=True)
    after_synthetic = len(df)
    logger.info(f"[Step 7] After adding synthetic samples: {after_synthetic} rows (added {after_synthetic - before_synthetic})")

    # Step 8: Oversample spam class to balance dataset
    logger.info("[Step 8] Balancing dataset via oversampling...")
    spam_df = df[df['label'] == 1]
    ham_df = df[df['label'] == 0]

    # Target spam ratio: ~40% (67% of ham count)
    target_spam_count = int(len(ham_df) * 0.67)

    if len(spam_df) < target_spam_count:
        # Oversample spam class
        spam_oversampled = resample(
            spam_df,
            replace=True,
            n_samples=target_spam_count,
            random_state=42
        )
        df = pd.concat([ham_df, spam_oversampled], ignore_index=True)
        logger.info(f"[Step 8] Oversampled spam from {len(spam_df)} to {target_spam_count}")
    else:
        # Already balanced or spam is majority
        df = pd.concat([ham_df, spam_df], ignore_index=True)
        logger.info(f"[Step 8] No oversampling needed (spam: {len(spam_df)}, ham: {len(ham_df)})")

    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    logger.info(f"[Step 8] Dataset shuffled and reindexed")

    # Final summary with detailed stats
    total_rows = len(df)
    label_0_count = (df['label'] == 0).sum()
    label_1_count = (df['label'] == 1).sum()
    spam_ratio = label_1_count / total_rows * 100

    logger.info("=" * 50)
    logger.info("===== FINAL DATASET SUMMARY =====")
    logger.info(f"Total rows        : {total_rows}")
    logger.info(f"Ham  (label 0)    : {label_0_count}")
    logger.info(f"Spam (label 1)    : {label_1_count}")
    logger.info(f"Spam ratio        : {spam_ratio:.1f}%")
    logger.info(f"Synthetic ham added: 200")
    logger.info(f"Data retention    : {total_rows}/{total_raw} ({total_rows/total_raw*100:.1f}%) from raw")
    logger.info("=" * 50)

    return df


def encode_labels(df):
    """
    DEPRECATED: Use load_and_prepare_data() instead.
    Kept for backwards compatibility.
    """
    logger.warning("encode_labels() is deprecated. Use load_and_prepare_data() instead.")

    # Handle both string and numeric labels
    spam_variants = {"spam", "phishing", "smishing", "1"}
    ham_variants = {"ham", "legitimate", "safe", "0", "non-phishing", "non-spam"}

    def normalize_label(label):
        if isinstance(label, (int, float)):
            if label == 1 or label == 1.0:
                return 1
            elif label == 0 or label == 0.0:
                return 0
            # Multi-class phishing labels (>=2) map to spam (1)
            elif label >= 2:
                return 1
            return -1
        label_str = str(label).lower().strip()
        if label_str in spam_variants:
            return 1
        elif label_str in ham_variants:
            return 0
        # Check if string is a numeric >= 2 (for Mendeley multi-class)
        try:
            num_val = float(label_str)
            if num_val >= 2:
                return 1
        except ValueError:
            pass
        return -1

    df["label"] = df["label"].apply(normalize_label)
    df = df[df["label"] != -1]
    logger.info(f"Label distribution:\n{df['label'].value_counts()}")
    return df


def train_epoch(model, dataloader, optimizer, criterion, scheduler=None):
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

        # Step learning rate scheduler if provided
        if scheduler is not None:
            scheduler.step()

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


class EarlyStopping:
    """
    Early stopping to halt training when validation accuracy stops improving.

    Args:
        patience: Number of epochs to wait for improvement before stopping.
        min_delta: Minimum change to qualify as an improvement.
        restore_best: Whether to restore best checkpoint when stopping.
    """

    def __init__(self, patience=3, min_delta=0.0, restore_best=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.best_accuracy = 0.0
        self.best_epoch = 0
        self.best_state_dict = None
        self.epochs_without_improvement = 0
        self.should_stop = False

    def __call__(self, val_accuracy, epoch, model):
        """
        Check if training should stop.

        Args:
            val_accuracy: Current validation accuracy.
            epoch: Current epoch number (1-indexed).
            model: The model being trained.

        Returns:
            bool: True if this is a new best, False otherwise.
        """
        if val_accuracy > self.best_accuracy + self.min_delta:
            # New best - save checkpoint
            self.best_accuracy = val_accuracy
            self.best_epoch = epoch
            self.best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.epochs_without_improvement = 0
            return True
        else:
            self.epochs_without_improvement += 1
            if self.epochs_without_improvement >= self.patience:
                self.should_stop = True
            return False

    def restore_best_weights(self, model):
        """Restore the best checkpoint weights to the model."""
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)
            logger.info(f"Restored best weights from epoch {self.best_epoch} (accuracy: {self.best_accuracy:.4f})")


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

    # Load and prepare data using improved cleaning pipeline
    logger.info("Loading and preparing data from pipeline 1/")
    df = load_and_prepare_data()

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

    # Compute class weights to handle class imbalance
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.array([0, 1]),
        y=train_labels
    )
    weights_tensor = torch.FloatTensor(class_weights).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    logger.info(f"Class weights computed: [0: {class_weights[0]:.4f}, 1: {class_weights[1]:.4f}]")

    # Warmup scheduler for learning rate
    num_training_steps = len(train_dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=num_training_steps
    )
    logger.info(f"Warmup scheduler initialized: {WARMUP_STEPS} warmup steps, {num_training_steps} total steps")

    # Early stopping setup
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, min_delta=0.001, restore_best=True)

    # Ensure save directory exists
    os.makedirs(os.path.dirname(SAVE_MODEL_PATH), exist_ok=True)

    # Save tokenizer once at the start
    os.makedirs(SAVE_TOKENIZER_PATH, exist_ok=True)
    tokenizer.save_pretrained(SAVE_TOKENIZER_PATH)
    logger.info(f"Tokenizer saved to {SAVE_TOKENIZER_PATH}")

    # Training loop
    logger.info(f"Training for up to {EPOCHS} epochs (early stopping patience={early_stopping.patience})")

    for epoch in range(EPOCHS):
        logger.info(f"\nEpoch {epoch + 1}/{EPOCHS}")

        # Train
        train_loss = train_epoch(model, train_dataloader, optimizer, criterion, scheduler)
        logger.info(f"Training Loss: {train_loss:.4f}")

        # Validate
        val_accuracy = validate(model, val_dataloader)
        logger.info(f"Validation Accuracy: {val_accuracy:.4f}")

        # Check early stopping and save best checkpoint
        is_best = early_stopping(val_accuracy, epoch + 1, model)

        if is_best:
            # Save best model checkpoint
            torch.save(model.state_dict(), SAVE_MODEL_BEST_PATH)
            logger.info(f"✓ New best model saved to {SAVE_MODEL_BEST_PATH} (accuracy: {val_accuracy:.4f})")
        else:
            logger.info(f"  No improvement for {early_stopping.epochs_without_improvement}/{early_stopping.patience} epochs")

        # Check if we should stop
        if early_stopping.should_stop:
            logger.info(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break

    # Save last checkpoint (current state at end of training)
    torch.save(model.state_dict(), SAVE_MODEL_LAST_PATH)
    logger.info(f"Last checkpoint saved to {SAVE_MODEL_LAST_PATH}")

    # Restore best weights and copy to main model path for inference
    if early_stopping.restore_best and early_stopping.best_state_dict is not None:
        early_stopping.restore_best_weights(model)

    # Copy best model to text_model.pth for inference
    if os.path.exists(SAVE_MODEL_BEST_PATH):
        shutil.copy(SAVE_MODEL_BEST_PATH, SAVE_MODEL_PATH)
        logger.info(f"Best model copied to {SAVE_MODEL_PATH} for inference")

    # Training summary
    logger.info("=" * 60)
    logger.info("Training Summary:")
    logger.info(f"  Best epoch: {early_stopping.best_epoch}")
    logger.info(f"  Best validation accuracy: {early_stopping.best_accuracy:.4f}")
    logger.info(f"  Total epochs trained: {epoch + 1}")
    logger.info("=" * 60)

    # Run sanity checks
    if not sanity_check_text_model():
        logger.error("Training completed but sanity checks failed!")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Training completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
