from tkinter import Tk
from tkinter.filedialog import askopenfilename
from pyzbar.pyzbar import decode
import cv2
import numpy as np
import sys
from pylibdmtx.pylibdmtx import decode as dmtx_decode
from types import SimpleNamespace
import matplotlib.pyplot as plt

# Step 01: Image Acquisition & Preprocessing

def load_image():

    # Open file selection window
    Tk().withdraw()  # Hide Tkinter main window
    image_path = askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
    image = None

    if image_path:
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            print("Failed to load the image, check the format.")
        # else:
        #     # Show the origin image
        #     cv2.imshow("Origin Image", image)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()

    else:
        print("No image has been selected.")
    return image

def preprocessing(image):

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    # Apply thresholding to convert to binary
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return image

    # Select largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    # Calculate the minimum rectangle of the area
    rect = cv2.minAreaRect(largest_contour)
    angle = rect[-1]

    # Correct the rotation angle
    if angle < -45:
        angle += 90
    else:
        angle -= angle

    # Rotate the image if the angle is large enough
    if abs(angle) > 1:
        h, w = gray.shape[:2]
        center = (w//2, h//2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated

    return gray

# Step 02: Image Enhancement

def enhance_image(image):
    """Apply contrast enhancement techniques if needs"""

    # Check if image is a grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Power-Law Transformation
    mean_intensity = np.mean(image)
    if mean_intensity < 40:    # If image is dark, apply a gamma of less than 1
        gamma = 0.2
        gamma_corrected = np.array(255 * (image / 255) ** gamma, dtype=np.uint8)
    elif mean_intensity > 200:  # If image is light, apply a gamma of more than 1
        gamma = 2
        gamma_corrected = np.array(255 * (image / 255) ** gamma, dtype=np.uint8)
    else:
        gamma_corrected = image

    if decode(gamma_corrected):
        return gamma_corrected

    # Contrast Stretching
    k = 0.5
    alpha = 15
    normalized = gamma_corrected.astype(np.float32) / 255.0
    transformed = 1 / (1 + np.exp(-alpha * (normalized - k)))
    stretched = (transformed * 255).astype(np.uint8)

    variance = np.var(stretched) # Calculate local variance between pixels
    if variance < 10000:
        # Histogram Equalization
        equalized = cv2.equalizeHist(stretched)
    else:
        equalized = stretched

    # High Boost Filtering
    A = 1.7
    high_boost_kernel = np.array([[0, -1, 0], [-1, A+4, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(equalized, -1, high_boost_kernel)

    return sharpened

# Step 03: Image Restoration

def restore_image(image):
    """Apply noise reduction techniques if needs"""

    # Check if image is a grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Check if barcode exists before proceeding
    if decode(image):
        return image

    # Convert the image to the frequency domain (2D-DFT)
    fft = np.fft.fft2(image)
    fft_shifted = np.fft.fftshift(fft)

    rows, cols = image.shape
    crow, ccol = rows//2, cols//2  # Find the center of the image

    # Apply frequency analysis to detect any frequency noise in the image
    magnitude_spectrum = np.abs(fft_shifted)
    frequency_noise = np.sum(magnitude_spectrum[0:10, 0:10])  # Check for high frequency

    random_noise = np.mean(image < 20) + np.mean(image > 235)
    # Check if the image has random noise
    if random_noise > 0.1:

        mean_intensity = np.mean(image)
        if mean_intensity > 180:

            # Convert the image to the frequency domain (2D-DFT)
            fft = np.fft.fft2(image)
            fft_shifted = np.fft.fftshift(fft)

            rows, cols = image.shape
            crow, ccol = rows//2, cols//2  # Find the center of the image

            # Gaussian Low-Pass Filtering (GLPF)
            # Create GLPF
            D0 = 50  # Cutoff frequency
            y, x = np.ogrid[:rows, :cols]
            D = np.sqrt((x - ccol)**2 + (y - crow)**2)
            H = np.exp(-(D**2) / (2 * (D0**2)))  # GLPF equation

            # Apply GLPF
            filtered_dft = fft_shifted * H

            # Convert back to spatial domain
            fft_ishifted = np.fft.ifftshift(filtered_dft)
            filtered_image = np.fft.ifft2(fft_ishifted)
            filtered_image = np.abs(filtered_image).astype(np.uint8)

            return filtered_image

        # Median Filtering
        median_filtered = cv2.medianBlur(image, 3)
        return median_filtered

    if frequency_noise > 10000: # Notch Reject Filtering

        # Create notch filter mask
        x, y = np.meshgrid(np.arange(cols), np.arange(rows))
        mask = np.ones((rows, cols), np.float32)

        # Notch positions
        notches = [(crow - 30, ccol - 30), (crow + 30, ccol + 30), (crow - 30, ccol + 30), (crow + 30, ccol - 30)]
        for notch in notches:
            cy, cx = notch
            mask[cy - 5:cy + 5, cx - 5:cx + 5] = 0

        # Apply RNF
        filtered_dft = fft_shifted * mask

        # Convert back to spatial domain using inverse DFT
        fft_ishifted = np.fft.ifftshift(filtered_dft)
        img_filtered = np.abs(np.fft.ifft2(fft_ishifted))
        notch_filtered = img_filtered.astype(np.uint8)

        return notch_filtered

    return image

# Step 04: Morphological Processing

def morphological_processing(image):
    """Apply morphological transformations if needs"""

    # Check if barcode exists before proceeding
    if decode(image):
        return image

    # Closing operation (Vertical kernel)
    kernel_close = np.ones((3, 1), np.uint8)
    closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel_close)

    # Opening operation (Square kernel)
    kernel_open = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)

    # Thinning operation (Hit-or-Miss transformation)
    kernel_thin = np.array([[-1, -1, -1], [1, 1, -1], [-1, -1, -1]], dtype=np.uint8)
    hitmiss = cv2.morphologyEx(opened, cv2.MORPH_HITMISS, kernel_thin)
    thinned = image - hitmiss

    return thinned

# Step 05: Segmentation

def segment_barcode(image):
    """Extract barcode region if needs"""

    # Check if barcode exists before proceeding
    if decode(image):
        return image

    # Check if image is a grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find contours
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return image

    # Get largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    return image[y:y + h, x:x + w]

# Step 06: Representation and Description

def represent_barcode(image):

    # Apply threshold to convert the image to binary
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Check if barcode exists before proceeding
    if decode(binary):
        return binary

    # Sobel Gradient for edge detection
    kernelx = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    kernely = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobelx = cv2.filter2D(binary, -1, kernelx)  # Horizontal edges
    sobely = cv2.filter2D(binary, -1, kernely)  # Vertical edges
    gradient = abs(sobelx) + abs(sobely)

    # Combine image and edges
    combined = cv2.bitwise_or(image, gradient)

    return combined

# Step 07: Object Recognition

def decode_barcode(image):
    """Decode barcode using pyzbar or pylibdmtx libraries"""

    # Try with pyzbar library
    decoded = decode(image)

    # If pyzbar failed, try with pylibdmtx library
    if not decoded:
        decoded_dmtx = dmtx_decode(image)
        if decoded_dmtx:
            decoded = [SimpleNamespace(
                type='DataMatrix',
                data=d.data
            ) for d in decoded_dmtx]

    if decoded:
        print("\nDecoding Results:")
        print(f"Type: {decoded[0].type}")
        print(f"Data: {decoded[0].data.decode('utf-8')}")

    else:
        print("No barcode detected")

    return decoded

# Implement the program

def main():

    # Load image
    image = load_image()
    if image is None:
        sys.exit("No image selected")

    # Preprocessing
    preprocessed = preprocessing(image)

    # Processing pipeline
    enhanced = enhance_image(preprocessed)
    restored = restore_image(enhanced)
    if not decode(restored):
        restored = enhance_image(restored)
    morphological = morphological_processing(restored)
    segmented = segment_barcode(morphological)
    binary = represent_barcode(segmented)

    # Decoding
    decode_result = decode_barcode(binary)

    # Display results
    # cv2.imshow("Preprocessed Image", preprocessed)
    # cv2.imshow("Enhanced Image", enhanced)
    # cv2.imshow("Restored Image", restored)
    # cv2.imshow("Morphological Processing", morphological)
    # cv2.imshow("Segmented Barcode", segmented)
    # cv2.imshow("Final Image", binary)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Display results
    plt.figure(figsize=(12,6))
    plt.subplot(1,2, 1)
    plt.title("Origin Image")
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(1,2, 2)
    plt.title("Final Binary Image (Thresholding)")
    plt.imshow(binary, cmap='gray')
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    main()

