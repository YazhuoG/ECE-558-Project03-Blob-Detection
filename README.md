# Laplacian Blob Detector (LoG)

Scale-space blob detection using Laplacian of Gaussian (LoG) filtering with 3D non-maximum suppression.

## Overview

This project implements a complete blob detection system from scratch using scale-normalized Laplacian of Gaussian filters. The detector identifies circular blob-like structures across multiple scales in images, useful for feature detection, object recognition, and image analysis.

## Features

### Core Functionality
- **LoG Kernel Generation** - Mathematically correct Laplacian of Gaussian filters
- **Scale-Space Construction** - Multi-scale blob detection (σ₀, k, n_scales)
- **Scale Normalization** - Response = σ² × LoG for scale-invariant detection
- **3D Non-Maximum Suppression** - Custom implementation for peak detection
- **Automatic Circle Visualization** - Overlays detected blobs as circles (r = √2σ)

### Implementation Highlights
- **Dual Convolution Methods** - Spatial and FFT-based (using numpy.fft only)
- **Efficient Scale Space** - Optimized memory and computation
- **Batch Processing** - Process single images or entire folders
- **JSON Export** - Save detection data (position, scale, radius, response)
- **Synthetic Test Images** - Generate test cases with known blobs

## Project Structure

```
blob_detection/
├── Project03_Option2_LoG_Blob_Detector.ipynb  # Main notebook
├── test_implementation.py                       # Quick verification
├── run_on_test_images.py                       # Batch processing
├── optimize_parameters.py                       # Parameter tuning
├── find_best_threshold.py                      # Threshold optimization
├── fishes.jpg                                  # Test image 1
├── butterfly.jpg                               # Test image 2
├── sunflowers.jpg                              # Test image 3
├── outputs/                                    # Results directory
└── README.md                                   # This file
```

## Requirements

### Python Packages
```bash
pip install numpy matplotlib imageio pillow
```

### Optional
```bash
pip install jupyter  # For running notebooks
```

## Usage

### 1. Run Complete Notebook

```bash
jupyter notebook Project03_Option2_LoG_Blob_Detector.ipynb
```

Execute all cells top-to-bottom, or jump to Section 10 for one-click execution.

### 2. Quick Verification

```bash
python3 test_implementation.py
```

Runs unit tests on all core functions with synthetic data.

### 3. Process Test Images

```bash
python3 run_on_test_images.py
```

Processes all test images with optimized parameters.

### 4. Parameter Optimization

```bash
python3 optimize_parameters.py
python3 find_best_threshold.py
```

## API Reference

### Basic Usage

```python
from blob_detector import LoGBlobDetector

# Create detector
detector = LoGBlobDetector(
    sigma0=2.0,      # Initial scale
    k=1.3,           # Scale factor
    n_scales=15,     # Number of scales
    threshold=0.005  # Response threshold
)

# Detect blobs
image = load_image("fishes.jpg")
blobs = detector.detect(image)

# Visualize
detector.visualize(image, blobs, save_path="outputs/result.png")

# Save detections
detector.save_detections(blobs, "outputs/detections.json")
```

### Advanced Options

```python
# Use FFT convolution for speed
detector = LoGBlobDetector(..., use_fft=True)

# Adjust NMS neighborhood
detector = LoGBlobDetector(..., nms_size=3)  # 3×3×3 neighborhood

# Batch processing
detector.process_folder("images/", output_dir="outputs/")
```

## Algorithm Details

### 1. LoG Kernel Generation

```
LoG(x, y; σ) = -1/(πσ⁴) × [1 - (x² + y²)/(2σ²)] × exp(-(x² + y²)/(2σ²))
```

- Kernel size: 6σ + 1 (odd)
- Zero-sum property verified
- Normalized for scale invariance

### 2. Scale Space Construction

For each scale σᵢ = σ₀ × kⁱ:
```
Response(x, y, σᵢ) = σᵢ² × convolve(image, LoG(σᵢ))
```

Scale normalization ensures consistent blob detection across scales.

### 3. Non-Maximum Suppression

```python
def is_local_maximum(volume, y, x, s):
    """Check if point is local max in 3×3×3 neighborhood"""
    value = volume[y, x, s]
    neighborhood = volume[y-1:y+2, x-1:x+2, s-1:s+2]
    return value >= neighborhood.max() and value > threshold
```

### 4. Circle Radius Calculation

```
radius = √2 × σ
```

This radius corresponds to the blob's characteristic scale.

## Parameter Tuning Guide

### sigma0 (Initial Scale)
- **Small objects**: 2.0-3.0
- **Medium objects**: 3.0-5.0
- **Large objects**: 5.0-10.0

### k (Scale Factor)
- **Recommended**: 1.3
- **Finer sampling**: 1.2 (slower, more accurate)
- **Coarser sampling**: 1.5 (faster, may miss blobs)

### n_scales (Number of Scales)
- **Formula**: log(max_sigma/sigma0) / log(k)
- **Typical**: 12-15 scales
- More scales = larger blob range

### threshold (Response Threshold)
- **Strict** (fewer blobs): 0.01-0.02
- **Balanced**: 0.003-0.008
- **Loose** (more blobs): 0.001-0.003

### Tuning Strategy

1. **Too many false positives** → Increase threshold
2. **Missing large blobs** → Increase n_scales or sigma0
3. **Missing small blobs** → Decrease sigma0
4. **Blobs not detected** → Check scale range covers blob sizes

## Test Results

### Performance Benchmarks

| Image | Size | Scales | Blobs | Time |
|-------|------|--------|-------|------|
| fishes.jpg | 335×500 | 15 | 2,393 | 14.2s |
| butterfly.jpg | 356×493 | 12 | 1,995 | 14.6s |
| sunflowers.jpg | 357×328 | 14 | 2,616 | 13.7s |

**Total**: 7,004 blobs detected across all test images

### Output Files

For each image:
- `{name}_blobs.png` - Detected circles overlaid
- `{name}_blobs.json` - Detection data
- `{name}_maxproj.png` - Scale-space max projection

## Technical Details

### Convolution Methods

**Spatial Convolution**:
```python
# Direct implementation with padding
result = spatial_convolve2d(image, kernel, padding='reflect')
```

**FFT Convolution**:
```python
# Faster for large kernels
result = fft_convolve2d(image, kernel)  # Uses numpy.fft only
```

FFT method is ~10× faster for large images and kernels.

### Memory Optimization

- Store squared responses: `response_sq = (σ² × LoG)²`
- Better peak detection with squared values
- Efficient numpy array operations

### Edge Handling

- Reflect padding for convolution
- NMS ignores border pixels to avoid artifacts
- Configurable border width

## Academic Context

**Course**: ECE558 - Digital Image Processing
**Project**: Project03-Final - Option 2 (Blob Detection)
**Requirements**:
- No OpenCV/scikit-image blob detectors
- Custom convolution implementation
- Custom 3D NMS implementation
- Scale-normalized LoG filters
- Clean, reproducible notebook

## Compliance

✅ **All requirements met**:
- No built-in blob detectors used
- No gaussian_laplace, convolve2d, or ndimage filters
- Only numpy, matplotlib, imageio, pathlib, json, time, math
- Custom convolution (spatial + FFT)
- Custom 3D NMS
- Complete notebook with 11 sections (0-10)

## Validation

All implementations verified with:
1. LoG kernel sum ≈ 0 (zero-sum property)
2. Spatial vs FFT convolution match
3. Synthetic image detection accuracy
4. Visual verification on real images
5. JSON export/import correctness

## Known Limitations

- Processing time scales with image size × n_scales
- Very small blobs (σ < 1.0) may be missed
- Overlapping blobs can suppress each other in NMS
- Threshold is image-dependent (requires tuning)

## Future Enhancements

- [ ] Multi-threaded scale-space computation
- [ ] GPU acceleration with CuPy
- [ ] Adaptive threshold per image region
- [ ] Difference of Gaussians (DoG) alternative
- [ ] Color blob detection (per channel)
- [ ] Real-time video blob tracking

## References

- Lindeberg, T. (1998). "Feature Detection with Automatic Scale Selection"
- Lowe, D. (2004). "SIFT: Distinctive Image Features from Scale-Invariant Keypoints"
- ECE558 Course Notes - Scale-Space Theory

## License

Academic project - Educational use only

## Author

Yazhuo Gao

---

**Note**: For best results on new images, start with default parameters (σ₀=2.0, k=1.3, n_scales=15, threshold=0.005) and adjust based on the tuning guide above.
