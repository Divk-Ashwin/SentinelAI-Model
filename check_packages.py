#!/usr/bin/env python
"""Check if all required packages are installed."""
import sys

packages = {
    "torch": "PyTorch",
    "pandas": "Pandas",
    "numpy": "NumPy", 
    "transformers": "Transformers",
    "sklearn": "Scikit-learn",
    "tqdm": "tqdm",
    "nltk": "NLTK",
    "PIL": "Pillow",
    "cv2": "OpenCV",
    "fastapi": "FastAPI"
}

print("Checking installed packages...")
print("=" * 50)

missing = []
for module, name in packages.items():
    try:
        __import__(module)
        print(f"✓ {name:20} installed")
    except ImportError:
        print(f"✗ {name:20} MISSING")
        missing.append(name)

print("=" * 50)
if missing:
    print(f"\nMissing {len(missing)} packages: {', '.join(missing)}")
    sys.exit(1)
else:
    print("\nAll packages installed successfully!")
    sys.exit(0)
