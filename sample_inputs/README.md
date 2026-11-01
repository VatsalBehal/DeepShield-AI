# DeepShield AI — Sample Media Directory

This directory contains test images and video clips for evaluating image inference, video inference, and Streamlit upload workflows.

## Recommended Media Formats
- **Images:** `.jpg`, `.jpeg`, `.png`, `.webp` (Containing at least one clearly visible human face).
- **Videos:** `.mp4`, `.avi`, `.mov` (Containing a human subject with facial movement).

## Creating Smoke Test Samples
To automatically generate synthetic test inputs:
```powershell
python scripts/smoke_test.py
```
This generates:
- `sample_inputs/test_face.jpg`
- `sample_inputs/smoke_test_video.mp4`
