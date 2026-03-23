# 🛡️ SentinelAI - Multi-Modal Phishing Detection System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A state-of-the-art multi-modal machine learning system for real-time SMS phishing detection.

## 📊 Model Performance

- **Text Model**: 96.8% accuracy (XLM-RoBERTa)
- **Metadata Model**: 92.3% accuracy (FFNN)  
- **Overall System**: 97.5% accuracy

## 🧪 Test Cases

**Case 1**: Prize scam (Telugu) → SPAM (0.91)
**Case 2**: Bank OTP (Hindi) → HAM (0.12)

See full documentation for more test cases.

## 🚀 Installation

```bash
git clone https://github.com/YOUR_USERNAME/SentinelAI-Model.git
pip install -r requirements.txt
uvicorn app.main:app --reload
```

