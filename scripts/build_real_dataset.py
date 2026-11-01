"""
DeepShield AI - Dataset Generation from Real Human Face Media
Extracts real human portraits and creates paired deepfake manipulations with
facial boundary seams, color tone mismatches, and blending artifacts.
"""

import os
import random
import cv2
import numpy as np
from PIL import Image

def main():
    random.seed(42)
    np.random.seed(42)

    # Clean previous synthetic toy samples
    for split in ['train', 'val', 'test']:
        for cls in ['real', 'fake']:
            d = os.path.join('data', split, cls)
            os.makedirs(d, exist_ok=True)
            for f in os.listdir(d):
                if f.endswith(('.jpg', '.png')):
                    os.remove(os.path.join(d, f))

    # Load sliced human faces
    real_images = []
    for i in range(28):
        path = f'sample_inputs/grid_faces/face_{i:02d}.jpg'
        if os.path.exists(path):
            img = Image.open(path).convert('RGB')
            real_images.append(img)

    print(f"Loaded {len(real_images)} real human faces.")

    # Function to create realistic deepfake manipulation
    def create_deepfake_face(src_img, donor_img):
        w, h = src_img.size
        src_np = np.array(src_img)
        donor_np = np.array(donor_img)

        # Inner face region (eyes, nose, mouth)
        fx1, fy1 = int(w * 0.20), int(h * 0.20)
        fx2, fy2 = int(w * 0.80), int(h * 0.80)
        fw, fh = fx2 - fx1, fy2 - fy1

        donor_inner = cv2.resize(donor_np[fy1:fy2, fx1:fx2], (fw, fh))

        # Color tone disparity (typical deepfake color mismatch)
        mismatched_donor = np.clip(donor_inner.astype(np.float32) * [1.12, 0.94, 0.88], 0, 255).astype(np.uint8)

        # Elliptical soft mask for face blending
        mask = np.zeros((h, w), dtype=np.float32)
        center = (int((fx1 + fx2) / 2), int((fy1 + fy2) / 2))
        axes = (int(fw * 0.44), int(fh * 0.44))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (13, 13), 3.5)
        mask = np.expand_dims(mask, axis=2)

        # Donor face pasted onto source body
        inner_full = src_np.copy()
        inner_full[fy1:fy2, fx1:fx2] = mismatched_donor
        blended = (inner_full * mask + src_np * (1.0 - mask)).astype(np.uint8)

        # Visible boundary seam artifact (characteristic of face-swapping)
        cv2.ellipse(blended, center, axes, 0, 0, 360, (130, 160, 190), 1)

        return Image.fromarray(blended)

    fake_images = []
    n = len(real_images)
    for i in range(n):
        donor_idx = (i + 7) % n
        fake_images.append(create_deepfake_face(real_images[i], real_images[donor_idx]))

    # Split: 20 train, 4 val, 4 test
    indices = list(range(n))
    random.shuffle(indices)

    train_idx = indices[:20]
    val_idx = indices[20:24]
    test_idx = indices[24:]

    splits = [('train', train_idx), ('val', val_idx), ('test', test_idx)]
    for split_name, idx_list in splits:
        for idx in idx_list:
            real_images[idx].save(f'data/{split_name}/real/real_{idx:02d}.jpg', quality=95)
            fake_images[idx].save(f'data/{split_name}/fake/fake_{idx:02d}.jpg', quality=95)

    print("Dataset populated successfully!")
    for s in ['train', 'val', 'test']:
        r_count = len(os.listdir(f'data/{s}/real'))
        f_count = len(os.listdir(f'data/{s}/fake'))
        print(f"  {s.upper()}: {r_count} Real, {f_count} Fake")

if __name__ == '__main__':
    main()
