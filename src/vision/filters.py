"""
OmniHCI Studio - Computer Vision and Image Processing Engine
Comprehensive suite of point processing, spatial filtering, morphology, and frequency operations.
"""
import numpy as np
import cv2
from PIL import Image

def ensure_numpy(img) -> np.ndarray:
    """Ensure input is a numpy array in RGB format."""
    if isinstance(img, Image.Image):
        return np.array(img)
    if isinstance(img, np.ndarray):
        return img
    raise ValueError("Input must be a PIL Image or numpy array")

def ensure_gray(img: np.ndarray) -> np.ndarray:
    """Ensure image is single-channel grayscale uint8."""
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img

def to_grayscale(img) -> np.ndarray:
    arr = ensure_numpy(img)
    return ensure_gray(arr)

def negative_image(img) -> np.ndarray:
    arr = ensure_numpy(img)
    return 255 - arr

def threshold_binary(img, thresh: int = 127) -> np.ndarray:
    gray = to_grayscale(img)
    _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    return binary

def threshold_otsu(img) -> np.ndarray:
    gray = to_grayscale(img)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def threshold_adaptive(img, block_size: int = 11, c_val: int = 2) -> np.ndarray:
    gray = to_grayscale(img)
    if block_size % 2 == 0:
        block_size += 1
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c_val
    )

def contrast_stretch_linear(img) -> np.ndarray:
    arr = ensure_numpy(img).astype(np.float64)
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return arr.astype(np.uint8)
    stretched = (arr - mn) / (mx - mn) * 255.0
    return np.clip(stretched, 0, 255).astype(np.uint8)

def contrast_stretch_gamma(img, gamma: float = 0.5) -> np.ndarray:
    arr = ensure_numpy(img) / 255.0
    corrected = np.power(arr, gamma) * 255.0
    return np.clip(corrected, 0, 255).astype(np.uint8)

def contrast_stretch_log(img, c: float = 1.0) -> np.ndarray:
    arr = ensure_numpy(img).astype(np.float64)
    result = c * np.log1p(arr)
    if result.max() > 0:
        result = result / result.max() * 255.0
    return np.clip(result, 0, 255).astype(np.uint8)

def normalize_minmax(img) -> np.ndarray:
    return contrast_stretch_linear(img)

def normalize_zscore(img) -> np.ndarray:
    arr = ensure_numpy(img).astype(np.float64)
    mean, std = arr.mean(), arr.std() + 1e-9
    z = (arr - mean) / std
    z_min, z_max = z.min(), z.max()
    norm = (z - z_min) / (z_max - z_min + 1e-9) * 255.0
    return np.clip(norm, 0, 255).astype(np.uint8)

def gaussian_blur(img, ksize: int = 5, sigma: float = 0.0) -> np.ndarray:
    arr = ensure_numpy(img)
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(arr, (ksize, ksize), sigma)

def median_blur(img, ksize: int = 5) -> np.ndarray:
    arr = ensure_numpy(img)
    if ksize % 2 == 0:
        ksize += 1
    return cv2.medianBlur(arr, ksize)

def average_blur(img, ksize: int = 5) -> np.ndarray:
    arr = ensure_numpy(img)
    return cv2.blur(arr, (ksize, ksize))

def canny_edges(img, low_thresh: int = 50, high_thresh: int = 150) -> np.ndarray:
    gray = to_grayscale(img)
    return cv2.Canny(gray, low_thresh, high_thresh)

def sobel_edges(img, ksize: int = 3) -> np.ndarray:
    gray = to_grayscale(img)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    magnitude = np.sqrt(gx**2 + gy**2)
    return np.clip(magnitude / (magnitude.max() + 1e-9) * 255, 0, 255).astype(np.uint8)

def laplacian_edges(img, ksize: int = 3) -> np.ndarray:
    gray = to_grayscale(img)
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
    lap = np.abs(lap)
    return np.clip(lap / (lap.max() + 1e-9) * 255, 0, 255).astype(np.uint8)

def morph_operation(img, op_name: str = "dilation", ksize: int = 5) -> np.ndarray:
    arr = ensure_numpy(img)
    kernel = np.ones((ksize, ksize), np.uint8)
    op_lower = op_name.lower()
    
    if op_lower == "dilation":
        return cv2.dilate(arr, kernel)
    elif op_lower == "erosion":
        return cv2.erode(arr, kernel)
    elif op_lower == "opening":
        return cv2.morphologyEx(arr, cv2.MORPH_OPEN, kernel)
    elif op_lower == "closing":
        return cv2.morphologyEx(arr, cv2.MORPH_CLOSE, kernel)
    elif op_lower == "gradient":
        return cv2.morphologyEx(arr, cv2.MORPH_GRADIENT, kernel)
    elif op_lower == "tophat":
        return cv2.morphologyEx(arr, cv2.MORPH_TOPHAT, kernel)
    elif op_lower == "blackhat":
        return cv2.morphologyEx(arr, cv2.MORPH_BLACKHAT, kernel)
    return arr

def bit_plane_slice(img, bit: int = 7) -> np.ndarray:
    """Extract individual bit plane (0=LSB, 7=MSB)."""
    gray = to_grayscale(img)
    plane = (gray >> bit) & 1
    return (plane * 255).astype(np.uint8)

def reconstruct_from_msb(img, num_msb: int = 4) -> np.ndarray:
    """Reconstruct grayscale image using top N most significant bits."""
    gray = to_grayscale(img)
    mask = 0
    for b in range(8 - num_msb, 8):
        mask |= (1 << b)
    return (gray & mask).astype(np.uint8)

def apply_all_filters(img, op_name: str, **kwargs) -> np.ndarray:
    """Dispatcher function for Gradio interface."""
    arr = ensure_numpy(img)
    op = op_name.lower().strip()
    
    if "grayscale" in op:
        return to_grayscale(arr)
    elif "negative" in op:
        return negative_image(arr)
    elif "otsu" in op:
        return threshold_otsu(arr)
    elif "adaptive" in op:
        return threshold_adaptive(arr)
    elif "binary threshold" in op:
        return threshold_binary(arr, kwargs.get("thresh", 127))
    elif "linear contrast" in op:
        return contrast_stretch_linear(arr)
    elif "gamma" in op:
        return contrast_stretch_gamma(arr, kwargs.get("gamma", 0.5))
    elif "log" in op:
        return contrast_stretch_log(arr)
    elif "z-score" in op:
        return normalize_zscore(arr)
    elif "gaussian" in op:
        return gaussian_blur(arr, kwargs.get("ksize", 5))
    elif "median" in op:
        return median_blur(arr, kwargs.get("ksize", 5))
    elif "average" in op:
        return average_blur(arr, kwargs.get("ksize", 5))
    elif "canny" in op:
        return canny_edges(arr)
    elif "sobel" in op:
        return sobel_edges(arr)
    elif "laplacian" in op:
        return laplacian_edges(arr)
    elif "dilation" in op:
        return morph_operation(arr, "dilation")
    elif "erosion" in op:
        return morph_operation(arr, "erosion")
    elif "opening" in op:
        return morph_operation(arr, "opening")
    elif "closing" in op:
        return morph_operation(arr, "closing")
    elif "gradient" in op:
        return morph_operation(arr, "gradient")
    elif "bit plane msb" in op:
        return bit_plane_slice(arr, 7)
    elif "bit plane lsb" in op:
        return bit_plane_slice(arr, 0)
    elif "reconstruct 4 msb" in op:
        return reconstruct_from_msb(arr, 4)
    return arr
