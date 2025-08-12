#!/usr/bin/env python3
"""
Setup script for Kaggle environment.
Run this first in Kaggle to install required packages.
"""

import subprocess
import sys

def install_packages():
    """Install required packages for the resume NER training."""
    packages = [
        "seqeval>=1.2.2",
        "peft>=0.6.0", 
        "accelerate>=0.24.0",
        "transformers>=4.35.0",
        "datasets>=2.14.0",
        "rich>=13.0.0",
        "pyyaml>=6.0"
    ]
    
    print("Installing required packages for Resume NER training...")
    
    for package in packages:
        print(f"Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ Successfully installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            continue
    
    print("\n🎉 Setup complete! You can now run the training script.")
    print("Run: python -m src.train --config configs/train.yaml")

if __name__ == "__main__":
    install_packages()
