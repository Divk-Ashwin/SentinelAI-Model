"""
Hard Negatives Data Augmentation
Adds tricky legitimate messages (harsh/urgent but HAM) to fix false positives
"""
import os
import sys
import pandas as pd

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE_DIR = os.path.join(PROJECT_ROOT, "pipeline 1")
OUTPUT_CSV = os.path.join(PIPELINE_DIR, "hard_negatives.csv")

# Hard negative examples - Legitimate but harsh/urgent messages
# These are currently misclassified as SPAM by the model
HARD_NEGATIVES = [
    # === GOVERNMENT/LEGAL (Harsh but legitimate) ===
    "High Court Notice: Case filed against you for consumer complaint. Hearing on 15th April. Details: http://highcourt.gov.pk/case/2026/8821",
    "Court Summons: You are required to appear before District Court on 20-Mar-2026 at 10:00 AM. Case No: DC/2026/4521. Failure to appear may result in arrest warrant.",
    "GHMC Demolition Notice: Unauthorized construction detected at property #28/A/1023. Demolition scheduled in 30 days. Object at: ghmc.gov.in/objections",
    "Legal Notice: You have 7 days to respond to cease and desist letter sent via registered post. Non-compliance will result in court proceedings.",
    "Municipal Corporation: Your building violates fire safety norms. Occupancy certificate suspended. Rectify within 15 days or face sealing action.",

    # === TAX/FINANCIAL AUTHORITIES (Threatening but real) ===
    "Income Tax Dept: Unreported crypto gains of Rs.8,75,000 detected for FY 2025-26. File revised return within 15 days or face penalty under Section 271(1)(c). Login: incometax.gov.in",
    "GST Notice: Your monthly return GSTR-3B is overdue by 45 days. Late fee: Rs.10,000. File immediately at gst.gov.in to avoid prosecution.",
    "EPFO: Your UAN is not linked with Aadhaar. PF account will be frozen from 31-Mar-2026. Link immediately: epfindia.gov.in",
    "RBI Warning: Your bank account shows suspicious transaction pattern. Account temporarily frozen for verification. Visit branch with ID proof within 7 days.",
    "Customs Dept: Package #IN8821234 held at airport. Duty of Rs.15,000 payable. Non-payment within 10 days will result in auction.",

    # === EVICTION/PROPERTY (Harsh but legitimate) ===
    "Eviction Notice: Rent overdue by 4 months (Rs.80,000). Vacate premises within 15 days or face forceful eviction under Rent Control Act.",
    "Society Notice: Maintenance dues of Rs.45,000 pending since 6 months. Legal action will be initiated. Pay immediately to avoid litigation.",
    "Electricity Board: Your connection will be permanently disconnected due to theft of electricity detected by inspection. Show cause within 48 hours.",
    "Water Authority: Illegal boring detected at your property. Sealing action on 25-Mar-2026. Appeal: waterboard.gov.in",

    # === FAMILY EMERGENCIES (Money requests but genuine) ===
    "Mom here. Dad admitted to AIIMS emergency. Surgery needed urgently. Rs.50,000 required immediately. Send to my PhonePe 9876543210. Will return next week.",
    "Beta its Papa. Car accident. In hospital. Need Rs.75,000 for treatment now. My account blocked. Send to this number via UPI: 7788990011",
    "Sister calling from hospital reception. Delivery complications. Need Rs.1,00,000 for emergency C-section. Account details: HDFC 12345678901. Urgent please.",

    # === MEDICAL/HEALTH (Urgent but real) ===
    "Apollo Hospital: Your biopsy results show abnormal cells. Immediate consultation required with Dr. Kumar. Book emergency appointment: apollohospitals.com/emergency",
    "Blood Bank Urgent: Patient with rare blood group AB-ve needs immediate transfusion. Life threatening emergency. Contact: +91-9900112233",
    "COVID Test Positive: You tested positive for COVID-19. Isolate immediately for 7 days. Download certificate: cowin.gov.in Report contacts urgently.",

    # === EDUCATION (Harsh deadlines) ===
    "CBSE Exam Alert: Your admit card has discrepancies. Correction window closes today 6 PM. Login immediately: cbse.gov.in/students Exam on 25th March.",
    "University Notice: Your degree certificate on hold due to pending fine of Rs.8,500. Clear dues within 5 days or graduation will be cancelled.",
    "Scholarship Deadline: Your NSP scholarship verification pending. Submit Aadhaar within 48 hours or forfeit Rs.25,000 grant. Portal: scholarships.gov.in",

    # === INSURANCE/CLAIMS (Urgent but legitimate) ===
    "LIC: Your policy #12345678 lapsed due to non-payment of premium Rs.15,000. Grace period ends 31-Mar-2026. Revive immediately or lose coverage.",
    "Motor Insurance Expired: Your vehicle RC1234 insurance expired on 20-Mar-2026. Driving is illegal and punishable. Renew immediately or face prosecution.",
    "Health Insurance Claim Rejected: Your claim #CLM8821 for Rs.2,50,000 rejected due to pre-existing disease non-disclosure. Appeal within 30 days with medical records.",

    # === MULTILINGUAL HARSH BUT LEGIT ===
    # Hindi
    "आयकर विभाग: आपकी ITR में गड़बड़ी पाई गई है। 15 दिन में revised return file करें वरना Rs.50,000 penalty लगेगी। incometax.gov.in पर login करें।",
    "कोर्ट नोटिस: आपके खिलाफ consumer case दर्ज हुआ है। 10 अप्रैल को hearing है। District Court Delhi. गैर-हाजिरी पर warrant जारी होगा।",
    "EPFO चेतावनी: आपका UAN Aadhaar से link नहीं है। 31 मार्च तक link न किया तो PF account freeze हो जाएगा।",

    # Telugu
    "కోర్ట్ నోటీసు: మీ పై consumer complaint case ఫైల్ అయింది। 15 ఏప్రిల్ న hearing. హాజరు కాకపోతే warrant issue అవుతుంది।",
    "GHMC Notice: మీ property లో unauthorized construction detected. 30 రోజుల్లో demolition. Object చేయండి: ghmc.gov.in",
    "Income Tax: మీ crypto transactions లో unreported gains Rs.8,75,000. 15 రోజుల్లో revised return file చేయండి లేదా penalty.",

    # Urdu
    "Court Notice: آپ کے خلاف consumer case filed ہوا۔ Hearing 15 اپریل کو۔ Court میں حاضری لازمی ورنہ warrant issue ہوگا۔",
    "Tax Department: آپ کی income tax return میں discrepancy ہے۔ 15 دن میں revised return file کریں ورنہ penalty Rs.50,000۔",
    "UIDAI: آپ کا Aadhaar card services suspend کر دیا گیا biometric mismatch کی وجہ سے۔ UIDAI center visit کریں documents کے ساتھ۔",

    # Tamil
    "Court Notice: உங்களுக்கு எதிராக consumer case தாக்கல் செய்யப்பட்டது। Hearing 15 ஏப்ரல் அன்று। ஆஜராகவில்லை என்றால் warrant.",
    "GHMC: உங்கள் property ல் unauthorized construction கண்டுபிடிக்கப்பட்டது। 30 நாட்களில் demolition. Object: ghmc.gov.in",
    "Income Tax: உங்கள் crypto transaction ல் unreported gains Rs.8,75,000. 15 நாட்களில் revised return இல்லையென்றால் penalty.",

    # === SERVICE DISRUPTIONS (Harsh warnings) ===
    "Airtel: Your SIM will be permanently blocked due to 90 days inactivity. Recharge minimum Rs.100 within 48 hours to avoid disconnection.",
    "Bank Notice: Your account XX5678 will be converted to dormant due to no transactions in 24 months. Visit branch immediately with KYC documents.",
    "Credit Card Alert: Your card ending 4521 blocked due to 3 failed payment attempts. Clear outstanding Rs.45,000 within 7 days to avoid legal action.",

    # === PROFESSIONAL/WORK (Harsh but real) ===
    "HR Notice: Your employment is terminated effective immediately for misconduct. Final settlement will be processed. Handover company assets within 24 hours.",
    "TCS: Your offer letter #TC2026 will expire on 31-Mar-2026. Accept and upload documents at careers.tcs.com immediately or offer will be withdrawn.",
    "Performance Warning: Your Q4 metrics are below 40%. This is your final warning. Improvement required in 30 days or termination as per company policy.",

    # === UTILITY DISCONNECTIONS ===
    "Electricity Dept: Your connection #IN8821 will be permanently disconnected for outstanding dues Rs.25,000. Pay within 5 days to avoid legal action.",
    "Gas Authority: Your LPG connection suspended due to safety violation. Inspection mandatory before restoration. Book: indangas.gov.in",
    "Broadband: Your connection terminated due to bandwidth misuse. Rs.5,000 penalty payable. Clear dues to restore service.",

    # === VISA/IMMIGRATION (Harsh but legitimate) ===
    "Embassy Notice: Your visa application #VIS8821 rejected due to incomplete documents. Re-apply with correct papers within 30 days or file fee forfeited.",
    "Immigration Alert: Your passport renewal delayed due to police verification pending. Visit RPO immediately with NOC or passport will expire.",
]

def main():
    print("=" * 70)
    print("HARD NEGATIVES AUGMENTATION - FIXING FALSE POSITIVES")
    print("=" * 70)

    # Create DataFrame with label=0 (HAM) for all
    data = [{"text": text, "label": 0} for text in HARD_NEGATIVES]
    df = pd.DataFrame(data)

    # Save to CSV
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')

    print(f"\n[OK] Created {len(HARD_NEGATIVES)} hard negative examples")
    print(f"[OK] All labeled as HAM (label=0)")
    print(f"[OK] Saved to: {OUTPUT_CSV}")
    print("\nThese examples will teach the model:")
    print("  - Government notices are harsh but legitimate")
    print("  - Legal warnings use urgent language but are real")
    print("  - Tax/financial authorities sound threatening but are genuine")
    print("  - Family emergencies with money requests can be real")
    print("  - Service disconnections are harsh but from real providers")
    print("\n" + "=" * 70)
    print("Next step: Add this to training data and retrain text model")
    print("=" * 70)

if __name__ == "__main__":
    main()
