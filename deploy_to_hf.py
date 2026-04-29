#!/usr/bin/env python3
"""
Deploy IPL 2026 Prediction Dashboard to Hugging Face Spaces.

Usage:
    python deploy_to_hf.py --username YOUR_HF_USERNAME

Prerequisites:
    - pip install huggingface_hub
    - huggingface-cli login  (or set HF_TOKEN env var)
"""

import argparse
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi, create_repo
except ImportError:
    print("Install huggingface_hub: pip install huggingface_hub")
    sys.exit(1)


SPACE_NAME = "ipl-2026-predictions"
SPACE_DIR = Path(__file__).parent / "hf_space"


def deploy(username: str, token: str | None = None):
    api = HfApi(token=token)
    repo_id = f"{username}/{SPACE_NAME}"

    # Create or get the Space
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="static",
            exist_ok=True,
            token=token,
        )
        print(f"✅ Space created/exists: https://huggingface.co/spaces/{repo_id}")
    except Exception as e:
        print(f"⚠️  Could not create space: {e}")
        print("Trying to upload anyway...")

    # Upload the entire hf_space directory
    api.upload_folder(
        folder_path=str(SPACE_DIR),
        repo_id=repo_id,
        repo_type="space",
        token=token,
    )

    url = f"https://huggingface.co/spaces/{repo_id}"
    print(f"\n🚀 Dashboard deployed successfully!")
    print(f"🔗 Live URL: {url}")
    print(f"\nShare this link on Twitter/X!")
    return url


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy IPL dashboard to HF Spaces")
    parser.add_argument("--username", required=True, help="Your HuggingFace username")
    parser.add_argument("--token", default=None, help="HF API token (or use HF_TOKEN env)")
    args = parser.parse_args()
    deploy(args.username, args.token)
