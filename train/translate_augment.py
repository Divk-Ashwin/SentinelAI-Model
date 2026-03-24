"""
Data Augmentation: Translate English SMS to Hindi/Telugu/Urdu/Tamil for multilingual training
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

URDU_SPAM = [
    "آپ کا اکاؤنٹ بند ہو جائے گا، ابھی verify کریں",
    "مبارک ہو! آپ نے 50,000 روپے جیتے ہیں",
    "فوری کلک کریں ورنہ account block ہوگا",
    "KYC update کریں ورنہ service بند ہوگی",
    "آپ کا UPI block ہو گیا، فوری update کریں",
    "آپ کا account suspend ہو گیا، link پر click کریں",
    "Free gift claim کریں، offer صرف آج تک",
    "Lottery winner ہیں آپ، ابھی claim کریں",
    "Bank account verify کریں ورنہ بند ہوگا",
    "آپ کے account میں suspicious activity ملی",
    "TRAI: آپ کا number block ہوگا، verify کریں",
    "آپ کا Aadhaar link نہیں ہے، ابھی کریں",
    "Prize money claim کریں: http://win-prize.xyz",
    "آپ کے account سے unauthorized transaction ہوا",
    "SBI: آپ کا KYC expire ہو گیا، update کریں"
]

URDU_HAM = [
    "آپ کا OTP 847293 ہے۔ 10 منٹ میں expire ہوگا۔ کسی سے share نہ کریں",
    "آپ کے اکاؤنٹ سے Rs.500 debit ہوئے۔ Available balance: Rs.4500",
    "آپ کا Amazon order deliver ہو گیا۔ Review کریں",
    "HDFC Bank: آپ کا payment successful رہا",
    "آپ کا Flipkart order ship ہو گیا۔ Track کریں",
    "SBI: آپ کے اکاؤنٹ میں Rs.10000 credit ہوئے",
    "آپ کی EMI successfully deduct ہو گئی",
    "Paytm: Payment of Rs.200 received successfully",
    "آپ کا recharge successful ہو گیا",
    "IRCTC: آپ کی ticket confirm ہو گئی",
    "آپ کا Swiggy order راستے میں ہے",
    "Airtel: آپ کا bill Rs.299 due ہے",
    "LIC: آپ کی premium successfully deduct ہوئی",
    "NSDL: آپ کا PAN card dispatch ہو گیا",
    "Zomato: آپ کا order 10 منٹ میں آئے گا"
]

TAMIL_SPAM = [
    "உங்கள் கணக்கு நிறுத்தப்படும், உடனே verify செய்யுங்கள்",
    "வாழ்த்துக்கள்! நீங்கள் ரூ.50,000 வென்றீர்கள்",
    "உடனே click செய்யுங்கள் இல்லையேல் account block ஆகும்",
    "உங்கள் KYC update செய்யுங்கள் இல்லை service நிற்கும்",
    "உங்கள் வங்கி விவரங்களை verify செய்யுங்கள் இப்போதே",
    "Free gift claim செய்யுங்கள், offer இன்றுடன் முடியும்",
    "உங்கள் account ல் suspicious activity கண்டுபிடிக்கப்பட்டது",
    "Lottery winner நீங்களே, உடனே claim செய்யுங்கள்",
    "உங்கள் UPI block ஆனது, உடனே update செய்யுங்கள்",
    "உங்கள் account suspend ஆனது, link click செய்யுங்கள்",
    "TRAI: உங்கள் number block ஆகும், verify செய்யுங்கள்",
    "உங்கள் Aadhaar link இல்லை, இப்போதே செய்யுங்கள்",
    "Prize money claim செய்யுங்கள்: http://win-prize.xyz",
    "உங்கள் account இலிருந்து unauthorized transaction நடந்தது",
    "SBI: உங்கள் KYC expire ஆனது, update செய்யுங்கள்"
]

TAMIL_HAM = [
    "உங்கள் OTP 847293. 10 நிமிடங்களில் expire ஆகும். யாரிடமும் சொல்லாதீர்கள்",
    "உங்கள் கணக்கிலிருந்து Rs.500 debit ஆனது. Balance: Rs.4500",
    "உங்கள் Amazon order deliver ஆனது. Review செய்யுங்கள்",
    "HDFC Bank: உங்கள் payment successful ஆனது",
    "உங்கள் Flipkart order ship ஆனது. Track செய்யுங்கள்",
    "SBI: உங்கள் கணக்கில் Rs.10000 credit ஆனது",
    "உங்கள் EMI successfully deduct ஆனது",
    "Paytm: Rs.200 payment received successfully",
    "உங்கள் recharge successful ஆனது",
    "IRCTC: உங்கள் ticket confirm ஆனது",
    "உங்கள் Swiggy order வழியில் உள்ளது",
    "Airtel: உங்கள் bill Rs.299 due உள்ளது",
    "LIC: உங்கள் premium successfully deduct ஆனது",
    "NSDL: உங்கள் PAN card dispatch ஆனது",
    "Zomato: உங்கள் order 10 நிமிடங்களில் வரும்"
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

        # Handle both 'label' and 'labels' columns
        if "labels" in df.columns and "label" not in df.columns:
            df["label"] = df["labels"]

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
    print("\n[1/8] Translating spam to Hindi...")
    hindi_spam = translate_batch(spam_texts, "hi", label=1, language_name="Hindi")
    all_data.extend(hindi_spam)

    print("\n[2/8] Translating ham to Hindi...")
    hindi_ham = translate_batch(ham_texts, "hi", label=0, language_name="Hindi")
    all_data.extend(hindi_ham)

    # Translate to Telugu
    print("\n[3/8] Translating spam to Telugu...")
    telugu_spam = translate_batch(spam_texts, "te", label=1, language_name="Telugu")
    all_data.extend(telugu_spam)

    print("\n[4/8] Translating ham to Telugu...")
    telugu_ham = translate_batch(ham_texts, "te", label=0, language_name="Telugu")
    all_data.extend(telugu_ham)

    # Translate to Urdu
    print("\n[5/8] Translating spam to Urdu...")
    urdu_spam = translate_batch(spam_texts, "ur", label=1, language_name="Urdu")
    all_data.extend(urdu_spam)

    print("\n[6/8] Translating ham to Urdu...")
    urdu_ham = translate_batch(ham_texts, "ur", label=0, language_name="Urdu")
    all_data.extend(urdu_ham)

    # Translate to Tamil
    print("\n[7/8] Translating spam to Tamil...")
    tamil_spam = translate_batch(spam_texts, "ta", label=1, language_name="Tamil")
    all_data.extend(tamil_spam)

    print("\n[8/8] Translating ham to Tamil...")
    tamil_ham = translate_batch(ham_texts, "ta", label=0, language_name="Tamil")
    all_data.extend(tamil_ham)

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
    for text in URDU_SPAM:
        all_data.append({"text": text, "label": 1})
    for text in URDU_HAM:
        all_data.append({"text": text, "label": 0})
    for text in TAMIL_SPAM:
        all_data.append({"text": text, "label": 1})
    for text in TAMIL_HAM:
        all_data.append({"text": text, "label": 0})

    # Create DataFrame
    df = pd.DataFrame(all_data)

    # Calculate summary statistics
    initial_rows = len(df)
    hindi_spam_count = len(hindi_spam)
    hindi_ham_count = len(hindi_ham)
    telugu_spam_count = len(telugu_spam)
    telugu_ham_count = len(telugu_ham)
    urdu_spam_count = len(urdu_spam)
    urdu_ham_count = len(urdu_ham)
    tamil_spam_count = len(tamil_spam)
    tamil_ham_count = len(tamil_ham)
    hardcoded_count = len(HINDI_SPAM) + len(HINDI_HAM) + len(TELUGU_SPAM) + len(TELUGU_HAM) + len(URDU_SPAM) + len(URDU_HAM) + len(TAMIL_SPAM) + len(TAMIL_HAM)

    # Calculate rows dropped (failed translations)
    expected_translations = 500 * 8  # 500 each for Hindi/Telugu/Urdu/Tamil spam and ham
    actual_translations = hindi_spam_count + hindi_ham_count + telugu_spam_count + telugu_ham_count + urdu_spam_count + urdu_ham_count + tamil_spam_count + tamil_ham_count
    rows_dropped = expected_translations - actual_translations

    # Check if file exists to determine mode
    file_exists = os.path.exists(OUTPUT_CSV)

    if file_exists:
        # Append mode (don't write header)
        df.to_csv(OUTPUT_CSV, mode='a', header=False, index=False, encoding='utf-8')
        print("\nAppended to existing CSV file")
    else:
        # Write mode (include header)
        df.to_csv(OUTPUT_CSV, mode='w', header=True, index=False, encoding='utf-8')
        print("\nCreated new CSV file")

    # Print summary
    print("\n" + "=" * 50)
    print("TRANSLATION SUMMARY")
    print("=" * 50)
    print(f"Hindi spam translated  : {hindi_spam_count}")
    print(f"Hindi ham translated   : {hindi_ham_count}")
    print(f"Telugu spam translated : {telugu_spam_count}")
    print(f"Telugu ham translated  : {telugu_ham_count}")
    print(f"Urdu spam translated   : {urdu_spam_count}")
    print(f"Urdu ham translated    : {urdu_ham_count}")
    print(f"Tamil spam translated  : {tamil_spam_count}")
    print(f"Tamil ham translated   : {tamil_ham_count}")
    print(f"Hardcoded samples      : {hardcoded_count}")
    print(f"Rows dropped (failed)  : {rows_dropped}")
    print("=" * 50)
    print(f"Total rows saved       : {len(df)}")
    print("=" * 50)
    print(f"Saved to: {OUTPUT_CSV}")
    print("=" * 50)


if __name__ == "__main__":
    main()
