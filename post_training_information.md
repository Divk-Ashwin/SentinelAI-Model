# SentinelAI Post-Training Information

## 1. Executive Summary

This document captures the full project context, training plan, execution history, metrics, findings, and recommended next actions for the SentinelAI phishing/smishing detection system.

The system is a multi-modal classifier with three pipelines:
- Pipeline 1: Text model (XLM-RoBERTa)
- Pipeline 2: Image model (MobileNetV2)
- Pipeline 3: Metadata model (FFNN)

Final observed outcomes from training runs:
- Pipeline 1 (Text) best validation accuracy: 0.9946 at epoch 5 during a 15-epoch run
- Pipeline 3 (Metadata) best validation accuracy: 0.9800 at epoch 10 during a 10-epoch run

Important operational note:
- In API inference, text-only messages can be biased toward SPAM if text score is high, due to current fusion defaults and threshold.
- Metadata API loading currently has an architecture mismatch issue (details in Section 11).

---

## 2. Project Scope and Goal

### Goal
Build a production-ready backend that detects phishing/spam messages in real-time using multi-modal evidence:
- Text content intent and language patterns
- Image content and OCR signals
- URL, sender, and temporal metadata behavior

### High-level Decision Logic
- Weighted late fusion is used:
  - text: 0.5
  - metadata: 0.3
  - image: 0.2
- Global spam decision threshold: 0.6

---

## 3. Project Structure (Relevant)

- app/main.py: FastAPI application and prediction endpoint
- app/models.py: request and response schemas
- models/text_pipeline.py: text inference pipeline
- models/metadata_pipeline.py: metadata inference pipeline
- models/image_pipeline.py: image inference pipeline
- fusion/decision_fusion.py: weighted fusion module
- train/train_text_model.py: Pipeline 1 training script
- train/train_metadata_model.py: Pipeline 3 training script
- config.py: model config, training config, weights, threshold
- saved_models/: trained models and tokenizer/scaler artifacts
- pipeline 1/: text training datasets
- pipeline 3/: metadata training datasets

---

## 4. Data Sources and Coverage

### Pipeline 1 (Text)
Used files:
- Kaggle Multilingual Spam Data pipeline 1.csv
- Mendeley SMS Phishing Dataset pipeline 1.csv
- smishtank dataset pipeline 1.csv
- UCI SMS Spam Collection pipeline 1.csv

Observed in training logs:
- total rows loaded before cleaning: 18045
- rows after normalization and filtering: 6520
- split: 5216 train / 1304 validation

### Pipeline 3 (Metadata)
Used files:
- HuggingFace phishing-dataset (ealvaradob) pipeline 3.csv
- ISCX-URL-2016 Dataset (CIC) pipeline 3.csv
- PhiUSIIL_Phishing_URL_Dataset pipeline 3.csv

Observed in training logs:
- one HuggingFace file skipped due to missing label column
- total rows loaded: 272502
- rows after label cleaning: 250079
- split: 200063 train / 50016 validation

---

## 5. Environment and Dependency Notes

Configured environment:
- Python virtual environment in .venv
- Python version detected: 3.14.0

Dependency compatibility note:
- requirements.txt pins older versions such as torch==2.1.2
- Python 3.14 required newer compatible installs in practice
- training was completed successfully with compatible installed versions

---

## 6. Model Definitions

### Pipeline 1 Text Model
- Base encoder: xlm-roberta-base
- Classification head: dropout + linear binary classifier
- Sequence length used in training: 128
- Batch size: 16
- Learning rate: 2e-5

### Pipeline 3 Metadata Model (Training)
- Input features: 15 engineered metadata features
- FFNN architecture: 15 -> 64 -> 32 -> 16 -> 2
- Activation: ReLU
- Dropout: 0.3
- Batch size: 16
- Learning rate: 2e-5
- StandardScaler used and saved

---

## 7. Training Plan Executed

### Plan
1. Verify dataset readiness
2. Verify dependencies and environment
3. Train Pipeline 1 (text)
4. Train Pipeline 3 (metadata)
5. Compare baseline and extended-epoch runs
6. Validate inference behavior via API

### Execution highlights
- Initial dependency install issues were resolved in the venv
- Pipeline 3 trained successfully after fixing missing tldextract package
- Pipeline 1 and Pipeline 3 were retrained with higher epoch counts for improvement analysis

---

## 8. Pre-Training and Baseline Results

### Pipeline 1 baseline run (3 epochs)
- Epoch 1 val accuracy: 0.9854
- Epoch 2 val accuracy: 0.9847
- Epoch 3 val accuracy: 0.9893
- Baseline best: 0.9893

### Pipeline 3 baseline run (3 epochs)
- Final val accuracy: 0.9751
- Precision: 0.9715
- Recall: 0.9873
- F1: 0.9793

---

## 9. Post-Training Extended Runs and Results

### Pipeline 3 extended run (10 epochs)
Key final metrics:
- Best epoch: 10
- Accuracy: 0.9800
- Precision: 0.9809
- Recall: 0.9857
- F1: 0.9833

Improvement vs baseline (3 epochs):
- Accuracy: 0.9751 -> 0.9800 (+0.0049)
- F1: 0.9793 -> 0.9833 (+0.0040)

### Pipeline 1 extended run (15 epochs)
Validation accuracy by epoch:
- e1: 0.9816
- e2: 0.9479
- e3: 0.9908
- e4: 0.9916
- e5: 0.9946
- e6: 0.9931
- e7: 0.9923
- e8: 0.9931
- e9: 0.9900
- e10: 0.9923
- e11: 0.9931
- e12: 0.9916
- e13: 0.9908
- e14: 0.9916
- e15: 0.9916

Best model checkpoint:
- epoch 5 with val accuracy 0.9946

Final epoch metrics:
- epoch 15 val accuracy: 0.9916

Improvement vs baseline (3 epochs):
- Best accuracy: 0.9893 -> 0.9946 (+0.0053)

Interpretation:
- Performance improved with longer training
- Best generalization appears around epoch 5
- Later epochs fluctuate and do not consistently improve beyond peak

---

## 10. Saved Artifacts

### Pipeline 1
- saved_models/text_model_epoch_1.pth ... text_model_epoch_15.pth
- saved_models/text_model.pth (active latest pointer)
- saved_models/text_tokenizer/

### Pipeline 3
- saved_models/metadata_model_epoch_1.pth ... metadata_model_epoch_10.pth
- saved_models/metadata_model.pth
- saved_models/metadata_scaler.pkl

Operational action performed:
- text_model.pth was switched to epoch 5 checkpoint for production testing (best val accuracy checkpoint)

---

## 11. Inference Validation and Observed Behavior

### API health
- /health endpoint reachable and healthy during tests

### Quick sample tests (text-only)
Observed false positives on legitimate transactional messages, including:
- Amazon shipment-like notification classified as SPAM
- OTP message (HDFC-style) classified as SPAM

Root cause for text-only bias:
- For missing modalities, metadata and image default to 0.5
- Final score = 0.5 * text + 0.3 * 0.5 + 0.2 * 0.5 = 0.5 * text + 0.25
- With threshold 0.6, any text score > 0.7 is labeled SPAM
- High text score can dominate and trigger SPAM for benign transactional text

Additional critical issue (metadata API path):
- Metadata model loading in API reported state_dict mismatch due to architecture mismatch between training model and inference model definition
- Result: metadata model may load as untrained/random in API process if not corrected

---

## 12. Risk Assessment

### Strengths
- Strong text validation performance on held-out split
- Improved metadata metrics with longer training
- Full training pipeline operational and reproducible

### Risks
- Real-world false positives for benign transactional messages
- Potential metadata API mismatch reducing multimodal reliability
- requirements.txt pinning may not be directly reproducible on Python 3.14

---

## 13. Recommended Next Steps

1. Fix metadata inference architecture in models/metadata_pipeline.py to exactly match training architecture used in train/train_metadata_model.py.
2. Add early stopping and best-checkpoint restore in text training to preserve best epoch automatically.
3. Recalibrate decision threshold and/or modality weights with a validation set focused on real transactional OTP/order notifications.
4. Add a hard negative set of benign transactional templates (OTP, delivery, billing reminders) to reduce false positives.
5. Add a post-training evaluation script that outputs confusion matrix, precision, recall, and F1 for Pipeline 1 (currently only val accuracy is logged).
6. Freeze reproducible dependency constraints for the active Python version.

---

## 14. Reproducibility Commands

Train text model:
- python train/train_text_model.py

Train metadata model:
- python train/train_metadata_model.py

Run API:
- python app/main.py

Health check:
- GET /health

Prediction:
- POST /predict with text and/or metadata and/or image payload

---

## 15. Current Status Snapshot

- Text model: trained through epoch 15, best checkpoint at epoch 5
- Metadata model: trained through epoch 10, best checkpoint at epoch 10
- API: running and reachable in local environment during validation
- Deployment readiness: good for internal testing; requires threshold calibration and metadata architecture alignment before production release

---

Document generated: 2026-03-19
Project: Model for SentinelAI
x