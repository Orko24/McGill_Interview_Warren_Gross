#!/usr/bin/env python3
"""
One-time setup: Read .env and create Modal secret from it
Run: python setup_modal.py
"""

import subprocess
import sys
from pathlib import Path

def main():
    env_file = Path(__file__).parent / ".env"
    
    if not env_file.exists():
        print("❌ .env file not found!")
        print("   Create it first: cp env.example .env")
        sys.exit(1)
    
    # Read HF_TOKEN from .env
    hf_token = None
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith("HF_TOKEN=") and not line.startswith("#"):
                hf_token = line.split("=", 1)[1].strip().strip('"').strip("'")
                break
    
    if not hf_token or hf_token.startswith("hf_xxx"):
        print("❌ HF_TOKEN not set in .env (or still placeholder)")
        print("   Edit .env and add your real token")
        sys.exit(1)
    
    print(f"✓ Found HF_TOKEN: {hf_token[:10]}...")
    
    # Create Modal secret
    print("Creating Modal secret 'huggingface'...")
    result = subprocess.run(
        ["modal", "secret", "create", "huggingface", f"HF_TOKEN={hf_token}"],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("✅ Modal secret 'huggingface' created!")
        print("\nNow run:")
        print("  modal run modal_app.py --quick --limit 50")
    else:
        if "already exists" in result.stderr.lower():
            print("⚠️  Secret 'huggingface' already exists. Updating...")
            # Delete and recreate
            subprocess.run(["modal", "secret", "delete", "huggingface", "-y"], capture_output=True)
            subprocess.run(["modal", "secret", "create", "huggingface", f"HF_TOKEN={hf_token}"])
            print("✅ Modal secret updated!")
        else:
            print(f"❌ Error: {result.stderr}")
            sys.exit(1)

if __name__ == "__main__":
    main()


