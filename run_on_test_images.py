#!/usr/bin/env python3
"""
Run blob detection on the actual test images.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json
import math
import time

try:
    import imageio.v3 as imageio
    IMAGE_BACKEND = 'imageio'
except ImportError:
    from PIL import Image
    IMAGE_BACKEND = 'PIL'

print(f"Using {IMAGE_BACKEND} for image I/O")

OUTPUT_DIR = Path('outputs')
OUTPUT_DIR.mkdir(exist_ok=True)

# Core functions
def log_kernel(sigma, size=None):
    if size is None:
        size = int(math.ceil(6 * sigma)) + 1
    if size % 2 == 0:
        size += 1
    center = size // 2
    y, x = np.ogrid[-center:center+1, -center:center+1]
    sigma_sq = sigma ** 2
    r_sq = x**2 + y**2
    kernel = -(1.0 / (np.pi * sigma_sq**2)) * \
             (1 - r_sq / (2 * sigma_sq)) * \
             np.exp(-r_sq / (2 * sigma_sq))
    kernel = kernel - kernel.mean()
    return kernel

def fft_convolve2d(image, kernel):
    h, w = image.shape
    kh, kw = kernel.shape
    fft_h = h + kh - 1
    fft_w = w + kw - 1
    kernel_padded = np.zeros((fft_h, fft_w), dtype=np.float32)
    kh_half = kh // 2
    kw_half = kw // 2
    kernel_padded[:kh, :kw] = kernel
    kernel_padded = np.roll(kernel_padded, -kh_half, axis=0)
    kernel_padded = np.roll(kernel_padded, -kw_half, axis=1)
    image_padded = np.zeros((fft_h, fft_w), dtype=np.float32)
    image_padded[:h, :w] = image
    image_fft = np.fft.fft2(image_padded)
    kernel_fft = np.fft.fft2(kernel_padded)
    result_fft = image_fft * kernel_fft
    result = np.fft.ifft2(result_fft).real
    start_h = kh // 2
    start_w = kw // 2
    result = result[start_h:start_h+h, start_w:start_w+w]
    return result

def load_image(image_path):
    if IMAGE_BACKEND == 'imageio':
        img = imageio.imread(image_path)
    else:
        img = np.array(Image.open(image_path))
    img = img.astype(np.float32)
    if img.ndim == 3:
        img = 0.2989 * img[..., 0] + 0.5870 * img[..., 1] + 0.1140 * img[..., 2]
    if img.max() > 1.0:
        img = img / 255.0
    return img

def build_scale_space(image, sigma0=2.0, k=1.3, n_scales=10):
    h, w = image.shape
    scale_space = np.zeros((h, w, n_scales), dtype=np.float32)
    sigma_list = []
    print(f"  Building scale space with {n_scales} scales...")
    for i in range(n_scales):
        sigma = sigma0 * (k ** i)
        sigma_list.append(sigma)
        kernel = log_kernel(sigma)
        response = fft_convolve2d(image, kernel)
        response_normalized = (sigma ** 2) * response
        scale_space[:, :, i] = response_normalized ** 2
        if (i + 1) % 3 == 0 or i == n_scales - 1:
            print(f"    Scale {i+1}/{n_scales}: sigma={sigma:.2f}")
    return scale_space, sigma_list

def nms_3d(volume, sigma_list, threshold=0.005, nms_radius=1):
    h, w, n_scales = volume.shape
    detections = []
    print(f"  Performing 3D NMS (threshold={threshold:.4f})...")
    for s in range(n_scales):
        for y in range(h):
            for x in range(w):
                response = volume[y, x, s]
                if response < threshold:
                    continue
                is_max = True
                y_min = max(0, y - nms_radius)
                y_max = min(h, y + nms_radius + 1)
                x_min = max(0, x - nms_radius)
                x_max = min(w, x + nms_radius + 1)
                s_min = max(0, s - nms_radius)
                s_max = min(n_scales, s + nms_radius + 1)
                neighborhood = volume[y_min:y_max, x_min:x_max, s_min:s_max]
                if response < neighborhood.max():
                    is_max = False
                elif response == neighborhood.max():
                    max_positions = np.argwhere(neighborhood == response)
                    if len(max_positions) > 1:
                        local_y, local_x, local_s = y - y_min, x - x_min, s - s_min
                        if not (max_positions[0] == [local_y, local_x, local_s]).all():
                            is_max = False
                if is_max:
                    sigma = sigma_list[s]
                    detections.append((y, x, s, sigma, response))
    return detections

def detect_and_save(image_path, sigma0, k, n_scales, threshold):
    print(f"\n{'='*70}")
    print(f"Processing: {image_path}")
    print(f"{'='*70}")

    start_time = time.time()
    img = load_image(image_path)
    print(f"  Loaded image: {img.shape}")

    scale_space, sigma_list = build_scale_space(img, sigma0, k, n_scales)
    detections = nms_3d(scale_space, sigma_list, threshold)

    print(f"  Found {len(detections)} blobs")

    # Visualize
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(img, cmap='gray')
    ax.set_title(f'Detected Blobs (n={len(detections)})', fontsize=16)
    ax.axis('off')

    for y, x, s, sigma, response in detections:
        radius = np.sqrt(2) * sigma
        circle = plt.Circle((x, y), radius, color='red', fill=False,
                           linewidth=2, alpha=0.8)
        ax.add_patch(circle)

    # Save
    base_name = Path(image_path).stem
    overlay_path = OUTPUT_DIR / f'{base_name}_blobs.png'
    fig.savefig(overlay_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved overlay: {overlay_path}")

    # Save detections JSON
    json_data = [
        {
            'y': int(y), 'x': int(x), 'scale_index': int(s),
            'sigma': float(sigma), 'response': float(response),
            'radius': float(np.sqrt(2) * sigma)
        }
        for y, x, s, sigma, response in detections
    ]
    json_path = OUTPUT_DIR / f'{base_name}_blobs.json'
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"  Saved detections: {json_path}")

    # Save max projection
    max_proj = scale_space.max(axis=2)
    fig_proj, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(max_proj, cmap='hot')
    ax.set_title('Max Projection over Scales')
    ax.axis('off')
    maxproj_path = OUTPUT_DIR / f'{base_name}_maxproj.png'
    fig_proj.savefig(maxproj_path, dpi=150, bbox_inches='tight')
    plt.close(fig_proj)
    print(f"  Saved max projection: {maxproj_path}")

    elapsed = time.time() - start_time
    print(f"  Total time: {elapsed:.2f}s")

    return len(detections)

# Main execution
print("="*70)
print("RUNNING BLOB DETECTION ON TEST IMAGES")
print("="*70)

test_images = [
    ('fishes.jpg', 2.0, 1.3, 15, 0.005),
    ('butterfly.jpg', 4.0, 1.3, 12, 0.003),
    ('sunflowers.jpg', 3.0, 1.3, 14, 0.002),
]

total_blobs = 0
for img_path, sigma0, k, n_scales, threshold in test_images:
    if Path(img_path).exists():
        try:
            n = detect_and_save(img_path, sigma0, k, n_scales, threshold)
            total_blobs += n
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\nSkipping {img_path} (not found)")

print(f"\n{'='*70}")
print(f"PROCESSING COMPLETE!")
print(f"{'='*70}")
print(f"Total blobs detected across all images: {total_blobs}")
print(f"All outputs saved to: {OUTPUT_DIR.absolute()}")
