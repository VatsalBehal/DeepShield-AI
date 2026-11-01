"""
DeepShield AI - Environment and Hardware Diagnostic Tool
Validates Python runtime, GPU availability, installed packages,
directory structures, and face detection cascades.
"""

import os
import sys
import platform
import shutil

def check_system():
    print("=" * 65)
    print("      DEEPSHIELD AI - SYSTEM AND ENVIRONMENT DIAGNOSTIC       ")
    print("=" * 65)

    # 1. Operating System & Python Version
    py_ver = sys.version.split()[0]
    os_name = platform.system()
    os_ver = platform.release()
    print(f"OS:                 {os_name} {os_ver} ({platform.architecture()[0]})")
    print(f"Python Version:     {py_ver} (Target: >= 3.10)")

    major, minor = sys.version_info.major, sys.version_info.minor
    if major == 3 and minor >= 10:
        print("  -> Status:        COMPATIBLE [OK]")
    else:
        print("  -> Status:        WARNING: Python 3.10 or 3.11 strongly recommended.")

    # 2. Disk Space
    total, used, free = shutil.disk_usage(".")
    print(f"Disk Free Space:    {free / (1024**3):.2f} GB")

    # 3. Core Libraries Check
    print("\n--- Dependencies Verification ---")
    packages = [
        ("torch", "PyTorch"),
        ("torchvision", "Torchvision"),
        ("cv2", "OpenCV"),
        ("PIL", "Pillow"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("sklearn", "Scikit-Learn"),
        ("matplotlib", "Matplotlib"),
        ("streamlit", "Streamlit"),
        ("yaml", "PyYAML"),
        ("pytest", "Pytest"),
    ]

    all_pkgs_ok = True
    for mod_name, label in packages:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "installed")
            print(f"  [x] {label:<14} Version: {ver}")
        except ImportError:
            print(f"  [ ] {label:<14} NOT FOUND! Run: pip install -r requirements.txt")
            all_pkgs_ok = False

    # 4. Hardware Acceleration (CUDA Check)
    print("\n--- Hardware Acceleration (GPU / CPU) ---")
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        if cuda_avail:
            device_name = torch.cuda.get_device_name(0)
            mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"CUDA Available:     YES [NVIDIA GPU]")
            print(f"GPU Model:          {device_name} ({mem_gb:.2f} GB VRAM)")
            print("Mode:               High-speed GPU Training & Accelerated Inference.")
        else:
            print("CUDA Available:     NO (CPU Mode)")
            print("Hardware Note:      Training will be performed via Google Colab (Free T4 GPU).")
            print("                    Local inference, evaluation, and Streamlit run on CPU.")
    except Exception as e:
        print(f"Hardware check error: {e}")

    # 5. Face Detector Cascades Check
    print("\n--- Computer Vision Cascade Assets ---")
    try:
        import cv2
        candidates = [
            os.path.join(getattr(cv2.data, "haarcascades", ""), "haarcascade_frontalface_default.xml"),
            os.path.abspath("configs/haarcascade_frontalface_default.xml")
        ]
        found_cascade = None
        for p in candidates:
            if os.path.exists(p):
                found_cascade = p
                break

        if found_cascade:
            print(f"  [x] Haar Cascade:   Found at {found_cascade}")
        else:
            print(f"  [ ] Haar Cascade:   NOT FOUND. Place haarcascade_frontalface_default.xml in configs/")
    except Exception as e:
        print(f"Cascade check error: {e}")

    # 6. Project Directory Structure
    print("\n--- Project Directory Integrity ---")
    expected_dirs = [
        "configs", "data", "models", "notebooks",
        "src", "scripts", "tests", "outputs", "docs"
    ]
    for d in expected_dirs:
        exists = os.path.isdir(d)
        status = "[x] Found" if exists else "[ ] Missing"
        print(f"  {status:<12} {d}/")

    print("\n" + "=" * 65)
    if all_pkgs_ok:
        print("DIAGNOSTIC VERDICT: System is fully configured and ready! [PASS]")
    else:
        print("DIAGNOSTIC VERDICT: Some packages are missing. Check requirements.txt.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    check_system()
