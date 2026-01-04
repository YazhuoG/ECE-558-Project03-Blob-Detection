#!/usr/bin/env python3
"""
Find the best threshold to match PDF sample (~50-150 blobs for butterfly).
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json
import math

try:
    import imageio.v3 as imageio
    IMAGE_BACKEND = 'imageio'
except ImportError:
    from PIL import Image
    IMAGE_BACKEND = 'PIL'

OUTPUT_DIR = Path('outputs')

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
    for i in range(n_scales):
        sigma = sigma0 * (k ** i)
        sigma_list.append(sigma)
        kernel = log_kernel(sigma)
        response = fft_convolve2d(image, kernel)
        response_normalized = (sigma ** 2) * response
        scale_space[:, :, i] = response_normalized ** 2
    return scale_space, sigma_list

def nms_3d(volume, sigma_list, threshold=0.005, nms_radius=1):
    h, w, n_scales = volume.shape
    detections = []
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

# Load and build scale space once
print("Loading butterfly image and building scale space...")
img = load_image('butterfly.jpg')
scale_space, sigma_list = build_scale_space(img, sigma0=4.0, k=1.3, n_scales=10)

# Analyze response distribution
print(f"\nScale space response statistics:")
print(f"  Min: {scale_space.min():.6f}")
print(f"  Max: {scale_space.max():.6f}")
print(f"  Mean: {scale_space.mean():.6f}")
print(f"  Median: {np.median(scale_space):.6f}")
print(f"  90th percentile: {np.percentile(scale_space, 90):.6f}")
print(f"  95th percentile: {np.percentile(scale_space, 95):.6f}")
print(f"  99th percentile: {np.percentile(scale_space, 99):.6f}")
print(f"  99.9th percentile: {np.percentile(scale_space, 99.9):.6f}")

# Try a range of aggressive thresholds
print(f"\n{'='*70}")
print("Testing different thresholds:")
print(f"{'='*70}")

thresholds_to_test = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]

results = []
for threshold in thresholds_to_test:
    detections = nms_3d(scale_space, sigma_list, threshold)
    n_blobs = len(detections)
    results.append((threshold, n_blobs))
    print(f"  Threshold {threshold:5.2f}: {n_blobs:4d} blobs")

# Find threshold that gives ~50-150 blobs
target_min, target_max = 50, 150
best_threshold = None
best_diff = float('inf')

for threshold, n_blobs in results:
    if target_min <= n_blobs <= target_max:
        diff = abs(n_blobs - (target_min + target_max) / 2)
        if diff < best_diff:
            best_diff = diff
            best_threshold = threshold

if best_threshold is None:
    # Find closest
    for threshold, n_blobs in results:
        diff = min(abs(n_blobs - target_min), abs(n_blobs - target_max))
        if diff < best_diff:
            best_diff = diff
            best_threshold = threshold

print(f"\n{'='*70}")
print(f"RECOMMENDATION:")
print(f"{'='*70}")
print(f"Best threshold for butterfly (targeting 50-150 blobs): {best_threshold}")
print(f"This gives: {[n for t, n in results if t == best_threshold][0]} blobs")

# Generate result with best threshold
print(f"\nGenerating final result with threshold={best_threshold}...")
detections = nms_3d(scale_space, sigma_list, best_threshold)

fig, ax = plt.subplots(1, 1, figsize=(12, 12))
ax.imshow(img, cmap='gray')
ax.set_title(f'Detected Blobs (n={len(detections)}, threshold={best_threshold})', fontsize=16)
ax.axis('off')

for y, x, s, sigma, response in detections:
    radius = np.sqrt(2) * sigma
    circle = plt.Circle((x, y), radius, color='red', fill=False,
                       linewidth=2, alpha=0.8)
    ax.add_patch(circle)

overlay_path = OUTPUT_DIR / 'butterfly_blobs_OPTIMIZED.png'
fig.savefig(overlay_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {overlay_path}")

# Save JSON
json_data = [
    {
        'y': int(y), 'x': int(x), 'scale_index': int(s),
        'sigma': float(sigma), 'response': float(response),
        'radius': float(np.sqrt(2) * sigma)
    }
    for y, x, s, sigma, response in detections
]
json_path = OUTPUT_DIR / 'butterfly_blobs_OPTIMIZED.json'
with open(json_path, 'w') as f:
    json.dump(json_data, f, indent=2)
print(f"Saved: {json_path}")
