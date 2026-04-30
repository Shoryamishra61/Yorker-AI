#!/bin/bash
# ═══════════════════════════════════════════════════
# Push IPL ML project to GitHub + HuggingFace Spaces
# ═══════════════════════════════════════════════════
#
# PREREQUISITES:
#   1. Create a GitHub repo named "IPL_ML" on github.com
#   2. Have a GitHub Personal Access Token (PAT) with repo access
#   3. Have a HuggingFace token from https://huggingface.co/settings/tokens
#
# USAGE:
#   chmod +x push_all.sh
#   ./push_all.sh
# ═══════════════════════════════════════════════════

set -e

echo "🏏 IPL 2026 ML Prediction Engine — Deployment Script"
echo "======================================================"
echo ""

# --- GITHUB ---
echo "📦 Step 1: Push to GitHub"
echo "-------------------------"
read -p "Enter your GitHub username: " GH_USER
read -sp "Enter your GitHub PAT (personal access token): " GH_TOKEN
echo ""

REPO_NAME="Yorker-AI"
GH_URL="https://${GH_USER}:${GH_TOKEN}@github.com/${GH_USER}/${REPO_NAME}.git"

# Add remote if not exists
git remote remove origin 2>/dev/null || true
git remote add origin "$GH_URL"
git branch -M main
git push -u origin main

echo "✅ Pushed to GitHub: https://github.com/${GH_USER}/${REPO_NAME}"
echo ""

# --- HUGGINGFACE ---
echo "🤗 Step 2: Deploy to HuggingFace Spaces"
echo "----------------------------------------"
read -sp "Enter your HuggingFace token: " HF_TOKEN
echo ""

python3 -c "
import os
os.environ['HF_TOKEN'] = '${HF_TOKEN}'
from huggingface_hub import HfApi, create_repo

api = HfApi(token='${HF_TOKEN}')
username = api.whoami()['name']
repo_id = f'{username}/ipl-2026-predictions'

create_repo(repo_id=repo_id, repo_type='space', space_sdk='static', exist_ok=True, token='${HF_TOKEN}')
api.upload_folder(folder_path='hf_space', repo_id=repo_id, repo_type='space', token='${HF_TOKEN}')

print(f'✅ Dashboard deployed: https://huggingface.co/spaces/{repo_id}')
print(f'')
print(f'🔗 Share this link in your tweets!')
print(f'   Replace [YOUR HF SPACES LINK] with:')
print(f'   https://huggingface.co/spaces/{repo_id}')
"

echo ""
echo "🎉 All done! Both GitHub and HuggingFace are live."
echo ""
echo "Next steps:"
echo "  1. Go to https://github.com/${GH_USER}/${REPO_NAME} — verify the repo"
echo "  2. Check the HuggingFace Space — verify the dashboard"
echo "  3. Start posting your Twitter thread tomorrow (Apr 30)!"
