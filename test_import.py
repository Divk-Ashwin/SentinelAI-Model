#!/usr/bin/env python
"""Test script to check if app can be imported and identify errors."""

import sys
import traceback

print("Python version:", sys.version)
print("Python executable:", sys.executable)
print("\n" + "="*60)
print("Attempting to import app.main...")
print("="*60 + "\n")

try:
    from app.main import app
    print("[SUCCESS] App imported successfully!")
    print(f"App type: {type(app)}")
    print(f"App routes: {len(app.routes)}")
except Exception as e:
    print("[ERROR] Failed to import app!")
    print(f"\nError type: {type(e).__name__}")
    print(f"Error message: {str(e)}")
    print("\n" + "="*60)
    print("Full traceback:")
    print("="*60)
    traceback.print_exc()
    sys.exit(1)
