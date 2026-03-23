"""
Data Augmentation: Translate English SMS to Hindi/Telugu for multilingual training
Reads first 500 spam + 500 ham from pipeline 1/, translates them, adds hardcoded samples
"""
import os
import sys
import pandas as pd
from tqdm import tqdm
from deep_translator import GoogleTranslator

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE_DIR = os.path.join(PROJECT_ROOT, "pipeline 1")
OUTPUT_CSV = os.path.join(PIPELINE_DIR, "translated_phishing.csv")

# Hardcoded samples
HINDI_SPAM = [
    "आपका खाता बंद हो जाएगा, अभी verify करें",
    "बधाई हो! आपने 50,000 रुपये जीते हैं",
    "तुरंत क्लिक करें वरना account block होगा",
    "KYC update करें नहीं तो service बंद होगी",
    "आपका account suspend हो गया, link पर click करें",
    "Free gift claim करें, offer सिर्फ आज तक",
    "आपके account में suspicious activity मिली",
    "Lottery winner हैं आप, अभी claim करें",
    "Bank account verify करें वरना बंद होगा",
    "आपका UPI block हो गया, तुरंत update करें",
    "आपका PAN card expire हो गया, update करें",
    "TRAI: आपका number block होगा, verify करें",
    "आपके account से unauthorized transaction हुआ",
    "Prize money claim करें: http://win-prize.xyz",
    "आपका Aadhaar link नहीं है, अभी करें"
]

HINDI_HAM = [
    "आपका OTP 847293 है। 10 मिनट में expire होगा। किसी से share न करें",
    "आपके खाते से Rs.500 debit हुए। Available balance: Rs.4500",
    "आपका Amazon order deliver हो गया। Review करें",
    "HDFC Bank: आपका payment successful रहा",
    "आपका Flipkart order ship हो गया। Track करें",
    "SBI: आपके खाते में Rs.10000 credit हुए",
    "आपकी EMI successfully deduct हो गई",
    "Paytm: Payment of Rs.200 received successfully",
    "आपका recharge successful हो गया",
    "IRCTC: आपकी ticket confirm हो गई",
    "आपका Swiggy order रास्ते में है",
    "Zomato: आपका order 10 मिनट में आएगा",
    "Airtel: आपका bill Rs.299 due है",
    "LIC: आपकी premium successfully deduct हुई",
    "NSDL: आपका PAN card dispatch हो गया"
]

TELUGU_SPAM = [
    "మీ ఖాతా నిలిపివేయబడుతుంది, వెంటనే verify చేయండి",
    "అభినందనలు! మీరు రూ.50,000 గెలుచుకున్నారు",
    "వెంటనే click చేయండి లేకుంటే account block అవుతుంది",
    "మీ KYC update చేయండి లేదా service ఆగిపోతుంది",
    "మీ బ్యాంక్ వివరాలు verify చేయండి ఇప్పుడే",
    "Free gift claim చేయండి, offer నేటితో ముగుస్తుంది",
    "మీ account లో suspicious activity కనుగొనబడింది",
    "Lottery winner మీరే, వెంటనే claim చేయండి",
    "మీ UPI block అయింది, వెంటనే update చేయండి",
    "మీ account suspend అయింది, link click చేయండి",
    "మీ PAN card expire అయింది, update చేయండి",
    "TRAI: మీ number block అవుతుంది, verify చేయండి",
    "మీ Aadhaar link లేదు, ఇప్పుడే చేయండి",
    "Prize money claim చేయండి: http://win-prize.xyz",
    "మీ account నుండి unauthorized transaction జరిగింది"
]

TELUGU_HAM = [
    "మీ OTP 847293. 10 నిమిషాలలో expire అవుతుంది. ఎవరికీ చెప్పవద్దు",
    "మీ ఖాతా నుండి Rs.500 debit అయింది. Balance: Rs.4500",
    "మీ Amazon order deliver అయింది. Review చేయండి",
    "HDFC Bank: మీ payment successful అయింది",
    "మీ Flipkart order ship అయింది. Track చేయండి",
    "SBI: మీ ఖాతాలో Rs.10000 credit అయింది",
    "మీ EMI successfully deduct అయింది",
    "Paytm: Rs.200 payment received successfully",
    "మీ recharge successful అయింది",
    "IRCTC: మీ ticket confirm అయింది",
    "మీ Swiggy order దారిలో ఉంది",
    "Zomato: మీ order 10 నిమిషాల్లో వస్తుంది",
    "Airtel: మీ bill Rs.299 due ఉంది",
    "LIC: మీ premium successfully deduct అయింది",
    "NSDL: మీ PAN card dispatch అయింది"
]


def load_english_data():
    """Load first 500 spam + 500 ham from pipeline 1 CSVs."""
    print("Loading English data from pipeline 1/...")

    # Load all CSVs
    csv_files = [
        "UCI SMS Spam Collection pipeline 1.csv",
        "Mendeley SMS Phishing Dataset pipeline 1.csv",
        "smishtank dataset pipeline 1.csv",
        "Kaggle Multilingual Spam Data pipeline 1.csv"
    ]

    spam_texts = []
    ham_texts = []

    for csv_file in csv_files:
        path = os.path.join(PIPELINE_DIR, csv_file)
        if not os.path.exists(path):
            continue

        # Try different encodings
        df = None
        for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
            try:
                df = pd.read_csv(path, encoding=encoding)
                break
            except:
                continue

        if df is None:
            continue

        # Normalize column names
        df.columns = df.columns.str.lower()

        # Extract label and text
        if "label" in df.columns and "text" in df.columns:
            # Normalize labels
            df["label"] = df["label"].astype(str).str.lower().str.strip()

            # Spam samples
            spam_mask = df["label"].isin(["1", "spam", "phishing", "smishing"])
            spam_texts.extend(df[spam_mask]["text"].dropna().astype(str).head(200).tolist())

            # Ham samples
            ham_mask = df["label"].isin(["0", "ham", "legitimate", "safe"])
            ham_texts.extend(df[ham_mask]["text"].dropna().astype(str).head(200).tolist())

    # Limit to 500 each
    spam_texts = spam_texts[:500]
    ham_texts = ham_texts[:500]

    print(f"Loaded {len(spam_texts)} spam and {len(ham_texts)} ham messages")
    return spam_texts, ham_texts


def translate_batch(texts, target_lang, label, language_name):
    """Translate a batch of texts to target language."""
    translator = GoogleTranslator(source='auto', target=target_lang)

    translated_data = []
    print(f"Translating to {language_name}...")

    for text in tqdm(texts, desc=f"{language_name} {label}"):
        try:
            # Limit text length to avoid API issues
            text_to_translate = str(text)[:500]
            translated = translator.translate(text_to_translate)
            translated_data.append({
                "text": translated,
                "label": label
            })
        except Exception as e:
            print(f"Translation error: {e}")
            # Skip failed translations
            continue

    return translated_data


def main():
    print("=" * 60)
    print("SMS Translation Augmentation Pipeline")
    print("=" * 60)

    # Load English data
    spam_texts, ham_texts = load_english_data()

    all_data = []

    # Translate to Hindi
    print("\n[1/4] Translating spam to Hindi...")
    hindi_spam = translate_batch(spam_texts, "hi", label=1, language_name="Hindi")
    all_data.extend(hindi_spam)

    print("\n[2/4] Translating ham to Hindi...")
    hindi_ham = translate_batch(ham_texts, "hi", label=0, language_name="Hindi")
    all_data.extend(hindi_ham)

    # Translate to Telugu
    print("\n[3/4] Translating spam to Telugu...")
    telugu_spam = translate_batch(spam_texts, "te", label=1, language_name="Telugu")
    all_data.extend(telugu_spam)

    print("\n[4/4] Translating ham to Telugu...")
    telugu_ham = translate_batch(ham_texts, "te", label=0, language_name="Telugu")
    all_data.extend(telugu_ham)

    # Add hardcoded samples
    print("\nAdding hardcoded samples...")
    for text in HINDI_SPAM:
        all_data.append({"text": text, "label": 1})
    for text in HINDI_HAM:
        all_data.append({"text": text, "label": 0})
    for text in TELUGU_SPAM:
        all_data.append({"text": text, "label": 1})
    for text in TELUGU_HAM:
        all_data.append({"text": text, "label": 0})

    # Create DataFrame and save
    df = pd.DataFrame(all_data)
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')

    print("=" * 60)
    print(f"Total translated samples: {len(df)}")
    print(f"Spam: {(df['label'] == 1).sum()}")
    print(f"Ham: {(df['label'] == 0).sum()}")
    print(f"Saved to: {OUTPUT_CSV}")
    print("=" * 60)


if __name__ == "__main__":
    main()
