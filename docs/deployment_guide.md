# DeepShield AI — Free Cloud Deployment Guide

This guide walks you through deploying **DeepShield AI** as a live, public website with a permanent shareable URL (like `https://deepshield-ai.streamlit.app`) for your resume, college presentation, and portfolio.

---

## 🌟 Method 1: Streamlit Community Cloud (Recommended — 100% Free)

Streamlit Community Cloud is the official hosting platform for Streamlit applications. It connects directly to your GitHub repository, provides continuous deployment, and runs your app 24/7 for free.

### Step 1: Create a GitHub Repository
1. Go to [https://github.com/](https://github.com/) and log in (or sign up for free).
2. Click the **`+`** icon in the top-right corner and select **New repository**.
3. Set the repository name to:
   ```text
   DeepShield-AI
   ```
4. Choose **Public** (required for free Streamlit Community Cloud hosting).
5. **Do NOT** check "Add a README file" (we already have a complete one).
6. Click **Create repository**.

---

### Step 2: Push Your Local Code to GitHub
Open PowerShell in `D:\DEEPFAKE_DETECTION` and run the following three commands:

```powershell
# 1. Rename branch to main
git branch -M main

# 2. Link your local project to your new GitHub repository
# (Replace <YOUR_GITHUB_USERNAME> with your actual GitHub username)
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/DeepShield-AI.git

# 3. Push all code and deployable model weights to GitHub
git push -u origin main
```

*(If prompted, sign in to your GitHub account in the browser pop-up).*

---

### Step 3: Launch on Streamlit Community Cloud
1. Go to [https://share.streamlit.io/](https://share.streamlit.io/)
2. Click **Continue with GitHub** and authorize Streamlit.
3. On your Streamlit Cloud dashboard, click the blue **New app** (or **Create app**) button.
4. Fill in the deployment details:
   - **Repository:** `<YOUR_GITHUB_USERNAME>/DeepShield-AI`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL:** (Optional) You can customize your URL handle, e.g. `deepshield-ai.streamlit.app`.
5. Click **Deploy!**

---

### What Happens Behind the Scenes?
- Streamlit Cloud provisions an Ubuntu Linux cloud container.
- It automatically installs system computer vision libraries from [`packages.txt`](file:///d:/DEEPFAKE_DETECTION/packages.txt) (`libgl1`, `libglib2.0-0`).
- It installs all Python packages from [`requirements.txt`](file:///d:/DEEPFAKE_DETECTION/requirements.txt).
- It loads `models/best_model.pth` and launches `app.py`.
- Within 2 to 3 minutes, your live web application will be accessible worldwide!

---

## 🚀 Method 2: Hugging Face Spaces (Alternative Free Host)

If you prefer Hugging Face Spaces:
1. Go to [https://huggingface.co/spaces](https://huggingface.co/spaces) and log in.
2. Click **Create new Space**.
3. Space name: `DeepShield-AI`.
4. License: `MIT`.
5. Space SDK: Select **Streamlit**.
6. Hardware: Free (CPU basic · 2 vCPU · 16 GB RAM).
7. Visibility: **Public**.
8. Click **Create Space**.
9. Connect your GitHub repository under Space Settings, or push your code directly to Hugging Face Git remote.

---

## ⚡ Method 3: Instant Live Link from Your Laptop (Temporary Sharing)

If you need a public URL in 30 seconds for a quick teacher demonstration without pushing to GitHub first, you can tunnel your local running Streamlit app using **Localtunnel** or **Cloudflare Tunnel**:

### Step 1: Start Streamlit locally
In PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

### Step 2: Open a new PowerShell window and expose via Localtunnel
```powershell
npx localtunnel --port 8501
```
It gives you an instant public URL (e.g. `https://fresh-foxes-jump.loca.lt`) that anyone on any device can open as long as your laptop is running.
