"""
Text Analysis Pipeline - Multilingual BERT for SMS Classification
Returns detected spam keywords, attention-based token contributions, and explanations.
"""
import os
import re
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass
import numpy as np
import logging

from config import TEXT_MODEL_CONFIG, DEVICE, TEXT_TOKENIZER_PATH, TEXT_MODEL_PATH
from utils.preprocessing import detect_spam_keywords

logger = logging.getLogger(__name__)


# Rule-based patterns for post-processing adjustments
SAFE_PATTERNS = [
    # English safe patterns
    r'\botp\b',
    r'one.time.pass',
    r'valid for \d+',
    r'do not share',
    r'don.t share',
    r'\bdelivered\b',
    r'out for delivery',
    r'order.*ship',
    r'has been shipped',
    r'payment.*received',
    r'debited.*account',
    r'credited.*account',
    r'available bal',
    r'transaction.*alert',
    r'\btrack.*order\b',
    r'expected delivery',
    r'bill.*generated',
    r'premium.*received',
    r'postpaid.*bill',
    r'due date',
    r'hallticket',
    r'hall ticket',
    r'examination.*date',
    r'jntuh|mgit|osmania',
    # Government/Official patterns
    r'acknowledgement.*no',
    r'itr.*filed',
    r'request.*received',
    r'processing time',
    r'status.*check',
    r'refund.*credit.*within',
    r'incometax\.gov',
    r'uidai\.gov',
    r'\.gov\.in',
    # Security notification patterns
    r'new sign.in.*google',
    r'was this you',
    r'no action needed',
    r'myaccount\.google',
    # Telugu safe patterns
    r'debit అయింది',
    r'credit అయింది',
    r'balance.*rs',
    r'upi ref',
    r'available balance',
    r'merchant.*swiggy|zomato|amazon',
    r'\bsbi\b.*account',
    r'నిమిషాలలో expire',
    r'చెప్పవద్దు',
    r'successful అయింది',
    r'బిల్.*rs',
    # Hindi safe patterns
    r'debit हुए',
    r'credit हुए',
    r'successful रहा',
    r'deliver हो गया',
    r'recharge.*हो गया',
    r'electricity.*connection',
    r'amount due',
    r'call 1912',
    r'units consumed',
    r'bescom|tneb|msedcl|tsspdcl|apspdcl',
]

PHISHING_PATTERNS = [
    r'click.*(here|now|link|below)',
    r'verify.*account',
    r'account.*suspend',
    r'urgent.*action.*required',
    r'win.*prize',
    r'won.*lottery',
    r'claim.*reward',
    r'expire.*\d+.*hour',
    r'confirm.*identity',
    r'update.*kyc',
]


def apply_rules(text: str, raw_score: float) -> Tuple[float, List[str]]:
    """
    Apply rule-based post-processing to adjust model predictions.
    Reduces false positives on legitimate messages (OTP, bank alerts, etc.)

    Args:
        text: Input message text
        raw_score: Raw spam probability from model (0.0 - 1.0)

    Returns:
        Tuple of (adjusted_score, rules_fired) where rules_fired is a list of matched pattern names
    """
    text_lower = text.lower()

    # Count pattern matches
    safe_hits = sum(
        1 for p in SAFE_PATTERNS
        if re.search(p, text_lower, re.IGNORECASE)
    )
    phishing_hits = sum(
        1 for p in PHISHING_PATTERNS
        if re.search(p, text_lower, re.IGNORECASE)
    )

    rules_fired = []

    print(f"Safe hits: {safe_hits}, Phishing hits: {phishing_hits}, Text: {text[:60]}")

    # Phishing AND safe patterns both present - Phishing always wins
    if phishing_hits >= 1 and safe_hits >= 1:
        rules_fired.append("phishing_priority")
        return raw_score, rules_fired

    # Clearly safe — cap at 0.30
    elif safe_hits >= 2 and phishing_hits == 0:
        rules_fired.append("safe_capped")
        return min(raw_score, 0.30), rules_fired

    # Clearly phishing — floor at 0.70 (even 1 phishing pattern with no safe patterns)
    elif phishing_hits >= 1 and safe_hits == 0:
        rules_fired.append("phishing_floored")
        return max(raw_score, 0.70), rules_fired

    # Unclear — use raw score
    else:
        return raw_score, rules_fired


@dataclass
class TokenContribution:
    """A token and its attention-based contribution score."""
    word: str
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return {"word": self.word, "score": round(self.score, 4)}


@dataclass
class TextAnalysisResult:
    """Complete result from text analysis with explainability."""
    score: float                              # Spam probability (after rules)
    raw_score: float                          # Raw model score (before rules)
    label: str                                # "SPAM" or "HAM"
    contributing_words: List[TokenContribution]  # Top tokens by attention
    detected_keywords: List[str]              # Rule-based keyword matches
    rules_fired: List[str]                    # Post-processing rules that fired
    explanation: str                          # Human-readable explanation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "raw_score": round(self.raw_score, 4),
            "label": self.label,
            "contributing_words": [t.to_dict() for t in self.contributing_words],
            "detected_keywords": self.detected_keywords,
            "rules_fired": self.rules_fired,
            "explanation": self.explanation
        }


class TextClassifier(nn.Module):
    """
    Multilingual BERT-based text classifier for phishing detection.
    Supports English, Hindi, Hinglish, Telugu, and other languages.
    Can output attention weights for explainability.
    """

    def __init__(
        self,
        model_name: str = TEXT_MODEL_CONFIG["model_name"],
        num_labels: int = TEXT_MODEL_CONFIG["num_labels"],
        dropout: float = TEXT_MODEL_CONFIG["dropout"]
    ):
        super(TextClassifier, self).__init__()

        # Load pretrained multilingual BERT
        self.bert = AutoModel.from_pretrained(model_name)

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        """
        Forward pass with optional attention output.

        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            output_attentions: If True, return attention weights

        Returns:
            Tuple of (logits, attentions) where attentions is None if not requested
        """
        # Get BERT outputs with optional attentions
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions
        )

        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]

        # Classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # Return attentions if requested
        attentions = outputs.attentions if output_attentions else None

        return logits, attentions


class TextPipeline:
    """
    Complete text analysis pipeline with attention-based explainability.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None
    ):
        """
        Initialize text pipeline.

        Args:
            model_path: Path to saved model weights
            tokenizer_path: Path to saved tokenizer
        """
        self.device = torch.device(DEVICE)
        self.max_length = TEXT_MODEL_CONFIG["max_length"]
        self.model_loaded = False

        # Determine tokenizer path
        if tokenizer_path is None:
            tokenizer_path = TEXT_TOKENIZER_PATH if TEXT_TOKENIZER_PATH else TEXT_MODEL_CONFIG["model_name"]

        # Load tokenizer (trained or pretrained)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
            logger.info(f"Loaded tokenizer from {tokenizer_path}")
        except Exception as e:
            logger.warning(f"Could not load tokenizer from {tokenizer_path}: {e}")
            logger.warning(f"Falling back to pretrained model: {TEXT_MODEL_CONFIG['model_name']}")
            self.tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_CONFIG["model_name"])

        # Initialize model
        self.model = TextClassifier()

        # Load trained weights if available
        model_path = model_path or TEXT_MODEL_PATH
        if model_path and os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                logger.info(f"✓ Loaded trained text model from {model_path}")
                self.model_loaded = True
            except Exception as e:
                logger.warning(f"✗ Could not load trained text model from {model_path}: {e}")
                logger.warning("Using pretrained XLM-RoBERTa base model (untrained classifier head)")
        elif model_path:
            logger.warning(f"✗ Text model file not found: {model_path}")
            logger.warning("Using pretrained XLM-RoBERTa base model (untrained classifier head)")

        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Preprocess text input.

        Args:
            text: Input SMS text

        Returns:
            Dictionary with input_ids and attention_mask
        """
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].to(self.device),
            'attention_mask': encoding['attention_mask'].to(self.device)
        }

    @torch.no_grad()
    def predict(self, text: str) -> float:
        """
        Predict phishing probability for text with rule-based post-processing.

        Args:
            text: Input SMS text

        Returns:
            Probability score (0.0 - 1.0) where higher = more likely spam
        """
        if not text or not text.strip():
            return 0.5  # Neutral score for empty text

        # Preprocess
        inputs = self.preprocess(text)

        # Forward pass (no attention needed for simple prediction)
        logits, _ = self.model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            output_attentions=False
        )

        # Get probability for SPAM class (assuming index 1)
        probs = torch.softmax(logits, dim=1)
        raw_spam_prob = probs[0, 1].item()

        # Apply rule-based post-processing
        adjusted_score, _ = apply_rules(text, raw_spam_prob)
        print(f"Raw: {raw_spam_prob:.4f} → Adjusted: {adjusted_score:.4f}")

        return adjusted_score

    @torch.no_grad()
    def _extract_attention_contributions(
        self,
        text: str,
        inputs: Dict[str, torch.Tensor],
        attentions: Tuple[torch.Tensor],
        top_k: int = 5
    ) -> List[TokenContribution]:
        """
        Extract top contributing tokens based on attention weights.

        Uses the last attention layer, averages across all heads,
        and focuses on attention from [CLS] token to other tokens.

        Args:
            text: Original input text
            inputs: Tokenized inputs
            attentions: Tuple of attention tensors from each layer
            top_k: Number of top tokens to return

        Returns:
            List of TokenContribution objects
        """
        if attentions is None or len(attentions) == 0:
            print(f"DEBUG: Early return - attentions is None: {attentions is None}, len: {len(attentions) if attentions else 'N/A'}")
            return []

        try:
            # Get last layer attention: shape (batch, heads, seq_len, seq_len)
            last_layer_attention = attentions[-1]

            # Average across all attention heads: shape (batch, seq_len, seq_len)
            avg_attention = last_layer_attention.mean(dim=1)

            # Get attention from CLS token (index 0) to all other tokens
            # Shape: (seq_len,)
            cls_attention = avg_attention[0, 0, :].cpu().numpy()

            # Get attention mask to identify valid tokens
            attention_mask = inputs['attention_mask'][0].cpu().numpy()

            # Get input tokens
            input_ids = inputs['input_ids'][0].cpu().numpy()
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

            # Build token-score pairs, excluding special tokens and padding
            token_scores = []
            for idx, (token, attn_score, mask) in enumerate(zip(tokens, cls_attention, attention_mask)):
                # Skip padding (mask=0), [CLS], [SEP], and other special tokens
                if mask == 0:
                    continue
                if token in ['<s>', '</s>', '<pad>', '[CLS]', '[SEP]', '[PAD]']:
                    continue
                if token.startswith('▁'):
                    # XLM-RoBERTa uses ▁ for word boundaries
                    token = token[1:]
                if not token or token.isspace():
                    continue

                token_scores.append((token, float(attn_score)))

            # Sort by attention score descending
            token_scores.sort(key=lambda x: x[1], reverse=True)

            # DEBUG: Show what we found
            print(f"DEBUG: Tokens found: {len(token_scores)}")
            print(f"DEBUG: Top tokens: {token_scores[:5]}")

            # Deduplicate tokens (keep highest score for each unique token)
            seen_tokens = set()
            unique_contributions = []
            for token, score in token_scores:
                token_lower = token.lower()
                if token_lower not in seen_tokens:
                    seen_tokens.add(token_lower)
                    unique_contributions.append(TokenContribution(word=token, score=score))
                if len(unique_contributions) >= top_k:
                    break

            # Normalize scores to 0-1 range
            if unique_contributions:
                max_score = max(c.score for c in unique_contributions)
                if max_score > 0:
                    for c in unique_contributions:
                        c.score = c.score / max_score

            print(f"DEBUG: Unique contributions: {len(unique_contributions)}")
            print(f"DEBUG: Contributions: {[(c.word, c.score) for c in unique_contributions]}")

            return unique_contributions

        except Exception as e:
            logger.warning(f"Failed to extract attention contributions: {e}")
            return []

    @torch.no_grad()
    def predict_with_explanation(
        self,
        text: str,
        top_k: int = 5
    ) -> TextAnalysisResult:
        """
        Predict with full explainability including attention-based token contributions.

        Args:
            text: Input SMS text
            top_k: Number of top contributing tokens to return

        Returns:
            TextAnalysisResult with score, contributing words, and explanation
        """
        print("TEXT PIPELINE METHOD CALLED: predict_with_explanation()")
        if not text or not text.strip():
            return TextAnalysisResult(
                score=0.5,
                raw_score=0.5,
                label="HAM",
                contributing_words=[],
                detected_keywords=[],
                rules_fired=[],
                explanation="Text is empty"
            )

        # Preprocess
        inputs = self.preprocess(text)

        # Forward pass WITH attention
        logits, attentions = self.model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            output_attentions=True
        )

        # Get raw probability
        probs = torch.softmax(logits, dim=1)
        raw_spam_prob = probs[0, 1].item()

        # Apply rule-based post-processing
        adjusted_spam_prob, rules_fired = apply_rules(text, raw_spam_prob)
        print(f"Raw: {raw_spam_prob:.4f} → Adjusted: {adjusted_spam_prob:.4f}")
        label = "SPAM" if adjusted_spam_prob > 0.5 else "HAM"

        # Extract attention-based contributions
        print(f"DEBUG: Attentions type: {type(attentions)}")
        print(f"DEBUG: Attentions is None: {attentions is None}")
        if attentions is not None:
            print(f"DEBUG: Attentions length: {len(attentions)}")
            if len(attentions) > 0:
                print(f"DEBUG: Last layer attention shape: {attentions[-1].shape}")

        contributing_words = self._extract_attention_contributions(
            text, inputs, attentions, top_k=top_k
        )

        # Detect rule-based keywords
        detected_keywords = detect_spam_keywords(text)

        # Generate explanation
        explanation = self._generate_explanation(adjusted_spam_prob, contributing_words, detected_keywords, rules_fired)

        return TextAnalysisResult(
            score=adjusted_spam_prob,
            raw_score=raw_spam_prob,
            label=label,
            contributing_words=contributing_words,
            detected_keywords=detected_keywords,
            rules_fired=rules_fired,
            explanation=explanation
        )

    @torch.no_grad()
    def predict_with_keywords(self, text: str) -> Tuple[float, List[str], str]:
        """
        Predict phishing probability and return detected keywords (Pipeline 1).
        Legacy method for backwards compatibility.

        Args:
            text: Input SMS text

        Returns:
            Tuple of (score, detected_keywords_list, explanation)
        """
        result = self.predict_with_explanation(text)
        return result.score, result.detected_keywords, result.explanation

    def _generate_explanation(
        self,
        score: float,
        contributing_words: List[TokenContribution],
        keywords: List[str],
        rules_fired: List[str] = []
    ) -> str:
        """Generate explanation for text-based prediction."""
        parts = []

        if score > 0.7:
            parts.append("High-risk spam detected")
        elif score > 0.5:
            parts.append("Moderate spam probability")
        else:
            parts.append("Text appears legitimate")

        # Add contributing words info
        if contributing_words:
            top_words = [c.word for c in contributing_words[:3]]
            parts.append(f"Key tokens: {', '.join(top_words)}")

        # Add keyword info
        if keywords:
            parts.append(f"Suspicious keywords: {', '.join(keywords[:3])}")

        # Add rule adjustments
        if rules_fired:
            if "phishing_priority" in rules_fired:
                parts.append("Rule: Phishing priority (mixed signals)")
            elif "safe_capped" in rules_fired:
                parts.append("Rule: Adjusted down (legitimate patterns)")
            elif "phishing_floored" in rules_fired:
                parts.append("Rule: Adjusted up (phishing patterns)")

        return " | ".join(parts)

    def save_model(self, path: str):
        """Save model weights."""
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")
