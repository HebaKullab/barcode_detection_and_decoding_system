# Barcode Detection & Decoding System

An intelligent barcode reading system built with **Python**, **OpenCV**, and **Pyzbar** that handles low-quality, noisy, or distorted barcode images through an adaptive image processing pipeline — applying only the corrections each image actually needs.

---

## Project Overview

Real-world barcode images are rarely perfect. This system tackles common degradation scenarios — poor lighting, blur, noise, skew, and low contrast — by analyzing each image and intelligently selecting the appropriate enhancement steps before decoding.

The result: accurate barcode reading across a wide range of image conditions, without wasting computation on unnecessary operations.

---

## Features

- **Adaptive preprocessing** — applies only what each image needs, skipping unnecessary steps
- **Orientation correction** — detects and corrects skewed/tilted barcodes automatically
- **Contrast enhancement** — contrast stretching, histogram equalization, power-law (gamma) transformation
- **Noise removal** — median filter (salt & pepper), Gaussian LPF (random noise), Notch Reject Filter (periodic noise)
- **Morphological processing** — closing, opening, and thinning for cleaner barcode lines
- **Edge & line analysis** — Sobel gradient for barcode extraction verification
- **Multi-format decoding** — supports CODE128, EAN13, EAN8, UPC, QR Code, Data Matrix, and more
- **Simple GUI** — Tkinter file picker with side-by-side before/after display via Matplotlib

---

## Technical Pipeline

```
Input Image (via Tkinter file dialog)
    │
    ▼
1. Grayscale Conversion + Orientation Correction
   └─ Thresholding → Largest contour → minAreaRect → Rotate if |angle| > 1°
    │
    ▼
2. Try Direct Decode (pyzbar) ──► Success → Output & display results ✓
    │ Fail
    ▼
3. Image Enhancement
   ├─ Power-Law Transformation (gamma < 1 if dark, gamma > 1 if bright)
   ├─ Try decode after gamma → if success, skip remaining steps
   ├─ Sigmoid Contrast Stretching (k=0.5, alpha=15)
   ├─ Histogram Equalization (if variance < 10000)
   └─ High Boost Filter (A = 1.7)
    │
    ▼
4. Image Restoration
   ├─ Try decode → if success, skip
   ├─ Random noise (> 10% extreme pixels)?
   │   ├─ Bright + noisy → Gaussian Low-Pass Filter (freq. domain, D0=50)
   │   └─ Dark + noisy   → Median Filter (3×3)
   └─ Frequency noise detected → Notch Reject Filter (2D-DFT + mask)
    │
    ▼
5. Re-enhance if still not decoded (second enhancement pass)
    │
    ▼
6. Morphological Processing
   ├─ Try decode → if success, skip
   ├─ Closing  (3×1 vertical kernel): fill gaps in barcode lines
   ├─ Opening  (3×3 square kernel):   remove small external noise
   └─ Thinning (Hit-or-Miss):         normalize line widths
    │
    ▼
7. Segmentation
   ├─ Try decode → if success, skip
   └─ Contour detection → largest bounding box → crop barcode region
    │
    ▼
8. Binarization & Edge Enhancement
   ├─ Otsu Thresholding → binary image
   ├─ Try decode → if success, return
   └─ Sobel gradient (H + V) merged with binary image
    │
    ▼
9. Decode with pyzbar / pylibdmtx
    │
    ▼
Output: Type + Data + Side-by-side display (Matplotlib)
```

---

## Requirements

```bash
pip install opencv-python numpy pyzbar pylibdmtx matplotlib
```

> On Linux, you may also need:
> ```bash
> sudo apt-get install libzbar0 libdmtx0b
> ```

> On Windows, `pylibdmtx` may require the [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe).

---

## Usage

```bash
python main.py
```

1. A file picker dialog opens — select any barcode image (`.png`, `.jpg`, `.jpeg`, `.bmp`)
2. The system processes the image through the adaptive pipeline
3. Console output shows the decoded type and data
4. A Matplotlib window displays the original image alongside the final binary result

**Console output example:**
```
Decoding Results:
Type: EAN13
Data: 6221007028024
```

If no image is selected:
```
No image has been selected.
```

---

## Project Structure

```
barcode-detection/
│
├── main.py          # Full pipeline: load → preprocess → enhance →
│                    # restore → morphology → segment → decode → display
│
├── report/
│   └── barcode_detection_report.pdf
│
├── presentation/
│   └── barcode_detection.pptx
│
└── README.md
```

---

## Key Functions

| Function | Description |
|---|---|
| `load_image()` | Opens Tkinter file dialog, loads image with OpenCV |
| `preprocessing(image)` | Grayscale conversion + skew correction via `minAreaRect` |
| `enhance_image(image)` | Gamma correction → contrast stretching → histogram EQ → high boost |
| `restore_image(image)` | Median / Gaussian LPF / Notch Reject based on noise analysis |
| `morphological_processing(image)` | Closing → Opening → Thinning (Hit-or-Miss) |
| `segment_barcode(image)` | Contour-based bounding box crop |
| `represent_barcode(image)` | Otsu thresholding + Sobel edge enhancement |
| `decode_barcode(image)` | pyzbar first, falls back to pylibdmtx for Data Matrix |
| `main()` | Orchestrates the full pipeline and Matplotlib display |

---

## Test Results

All 14 test cases decoded successfully:

| # | Image Type | Barcode Type | Decoded Data |
|---|---|---|---|
| 01 | Clear (binary) | CODE128 | 220212022 |
| 02 | Clear (RGB) | EAN13 | 0123456789128 |
| 03 | Dark | CODE128 | dynamsoft |
| 04 | Bright/overexposed | EAN13 | 0123456789128 |
| 05 | Blurred | EAN13 | 0076950450479 |
| 06 | Noisy | CODE128 | 302940588302 |
| 07 | Salt & pepper noise | CODE128 | 220212022 |
| 08 | Periodic noise | CODE128 | 123456789 |
| 09 | Clear (with text label) | CODE128 | C0016 |
| 10 | Noisy | EAN13 | 9780201379624 |
| 11 | Noisy | EAN8 | 50296583 |
| 12 | Natural (real product) | EAN13 | 6221007028024 |
| 13 | Natural (real product, angled) | EAN13 | 6223007947589 |
| 14 | No barcode present | — | No barcode detected |

---

## Libraries Used

| Library | Purpose |
|---|---|
| `opencv-python` | Image processing, transforms, morphology, contours |
| `numpy` | Array operations, FFT, frequency-domain filtering |
| `pyzbar` | Decoding 1D/2D barcodes (CODE128, EAN, UPC, QR…) |
| `pylibdmtx` | Fallback decoding for Data Matrix barcodes |
| `matplotlib` | Displaying original vs. processed image side by side |
| `tkinter` | Native OS file picker dialog |

---

## Future Improvements

- Real-time barcode scanning via webcam
- Batch processing for multiple images at once
- Export decoded results to CSV or JSON
- Deep learning-based barcode localization for heavily degraded images
- Packaging as a standalone desktop app (PyInstaller / Tkinter full UI)
