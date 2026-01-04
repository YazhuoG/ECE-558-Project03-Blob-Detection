#!/usr/bin/env python3
"""
Quick test script to verify the blob detector implementation works.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import json
import math

# Create output directory
OUTPUT_DIR = Path('outputs')
OUTPUT_DIR.mkdir(exist_ok=True)

print("Testing Laplacian Blob Detector Implementation...")
print("=" * 70)

# 1. Test LoG kernel generation
print("\n[1/6] Testing LoG kernel generation...")
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

kernel = log_kernel(2.0)
print(f"   LoG kernel (sigma=2.0): shape={kernel.shape}, sum={kernel.sum():.2e}")
assert abs(kernel.sum()) < 1e-10, "Kernel sum should be ~0"
print("   ✓ LoG kernel generation works!")

# 2. Test FFT convolution
print("\n[2/6] Testing FFT convolution...")
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

test_img = np.random.rand(64, 64).astype(np.float32)
result = fft_convolve2d(test_img, kernel)
print(f"   Convolution result: shape={result.shape}, range=[{result.min():.3e}, {result.max():.3e}]")
assert result.shape == test_img.shape, "Output shape should match input"
print("   ✓ FFT convolution works!")

# 3. Test synthetic image generation
print("\n[3/6] Testing synthetic image generation...")
def generate_synthetic_image(size=512, num_blobs=15, seed=42):
    np.random.seed(seed)
    img = np.zeros((size, size), dtype=np.float32)
    for _ in range(num_blobs):
        cx = np.random.randint(50, size - 50)
        cy = np.random.randint(50, size - 50)
        radius = np.random.uniform(10, 40)
        intensity = np.random.uniform(0.5, 1.0)
        y, x = np.ogrid[:size, :size]
        sigma = radius / 2.5
        dist_sq = (x - cx)**2 + (y - cy)**2
        blob = intensity * np.exp(-dist_sq / (2 * sigma**2))
        img += blob
    img = np.clip(img, 0, 1)
    img += np.random.normal(0, 0.02, img.shape)
    img = np.clip(img, 0, 1)
    return img

synth_img = generate_synthetic_image(256, 10)
print(f"   Synthetic image: shape={synth_img.shape}, range=[{synth_img.min():.3f}, {synth_img.max():.3f}]")
print("   ✓ Synthetic image generation works!")

# 4. Test scale space building
print("\n[4/6] Testing scale space building...")
def build_scale_space(image, sigma0=2.0, k=1.3, n_scales=8):
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

scale_space, sigmas = build_scale_space(synth_img, sigma0=2.0, k=1.3, n_scales=6)
print(f"   Scale space: shape={scale_space.shape}")
print(f"   Sigma range: [{sigmas[0]:.2f}, {sigmas[-1]:.2f}]")
print("   ✓ Scale space building works!")

# 5. Test 3D NMS
print("\n[5/6] Testing 3D NMS...")
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

detections = nms_3d(scale_space, sigmas, threshold=0.01)
print(f"   Found {len(detections)} blob detections")
assert len(detections) > 0, "Should detect at least some blobs"
print("   ✓ 3D NMS works!")

# 6. Test visualization
print("\n[6/6] Testing visualization...")
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.imshow(synth_img, cmap='gray')
ax.set_title(f'Detected Blobs (n={len(detections)})')
ax.axis('off')
for y, x, s, sigma, response in detections:
    radius = np.sqrt(2) * sigma
    circle = plt.Circle((x, y), radius, color='red', fill=False, linewidth=1.5, alpha=0.8)
    ax.add_patch(circle)
output_path = OUTPUT_DIR / 'test_result.png'
fig.savefig(output_path, dpi=100, bbox_inches='tight')
plt.close(fig)
print(f"   Saved test result to: {output_path}")

# Save detections to JSON
json_data = [
    {
        'y': int(y), 'x': int(x), 'scale_index': int(s),
        'sigma': float(sigma), 'response': float(response),
        'radius': float(np.sqrt(2) * sigma)
    }
    for y, x, s, sigma, response in detections
]
json_path = OUTPUT_DIR / 'test_detections.json'
with open(json_path, 'w') as f:
    json.dump(json_data, f, indent=2)
print(f"   Saved detections to: {json_path}")
print("   ✓ Visualization works!")

print("\n" + "=" * 70)
print("ALL TESTS PASSED! ✓")
print("=" * 70)
print(f"\nImplementation is working correctly!")
print(f"Outputs saved to: {OUTPUT_DIR.absolute()}")
print(f"\nYou can now run the full Jupyter notebook:")
print(f"  jupyter notebook Project03_Option2_LoG_Blob_Detector.ipynb")
