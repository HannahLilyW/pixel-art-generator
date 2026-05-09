from PIL import Image
import numpy as np
from sklearn.cluster import MiniBatchKMeans


def _srgb_to_lab(rgb_uint8: np.ndarray) -> np.ndarray:
    """Convert (N, 3) uint8 RGB to (N, 3) LAB."""
    rgb = rgb_uint8.astype(np.float64) / 255.0

    # Linearize sRGB gamma
    linear = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)

    # Linear RGB -> XYZ (D65)
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    xyz = linear @ M.T / np.array([0.95047, 1.00000, 1.08883])

    # XYZ -> LAB
    f = np.where(xyz > 0.008856, xyz ** (1 / 3), (903.3 * xyz + 16) / 116)
    L = 116 * f[:, 1] - 16
    a = 500 * (f[:, 0] - f[:, 1])
    b = 200 * (f[:, 1] - f[:, 2])
    return np.stack([L, a, b], axis=1)


def _lab_to_rgb_uint8(lab: np.ndarray) -> np.ndarray:
    """Convert (N, 3) LAB to (N, 3) uint8 RGB."""
    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
    fy = (L + 16) / 116
    fx = a / 500 + fy
    fz = fy - b / 200

    xr = np.where(fx ** 3 > 0.008856, fx ** 3, (116 * fx - 16) / 903.3)
    yr = np.where(L > 903.3 * 0.008856, ((L + 16) / 116) ** 3, L / 903.3)
    zr = np.where(fz ** 3 > 0.008856, fz ** 3, (116 * fz - 16) / 903.3)
    xyz = np.stack([xr, yr, zr], axis=1) * np.array([0.95047, 1.00000, 1.08883])

    # XYZ -> linear RGB
    M_inv = np.array([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252],
    ])
    linear = np.clip(xyz @ M_inv.T, 0, 1)

    # Apply sRGB gamma
    rgb = np.where(linear > 0.0031308, 1.055 * linear ** (1 / 2.4) - 0.055, 12.92 * linear)
    return np.clip(rgb * 255, 0, 255).astype(np.uint8)


def quantize_colors(image: Image.Image, num_colors: int) -> tuple[Image.Image, list[tuple[int, int, int]]]:
    """
    Reduce image to num_colors using k-means in perceptual LAB color space (no dithering).
    Returns the quantized RGB image and the palette as a list of (R, G, B) tuples.
    """
    rgb = image.convert("RGB")
    pixels = np.array(rgb).reshape(-1, 3)

    lab_pixels = _srgb_to_lab(pixels)

    kmeans = MiniBatchKMeans(n_clusters=num_colors, n_init=3, random_state=42, batch_size=4096)
    labels = kmeans.fit_predict(lab_pixels)

    centers_rgb = _lab_to_rgb_uint8(kmeans.cluster_centers_)
    quantized_pixels = centers_rgb[labels].reshape(np.array(rgb).shape)

    palette = [tuple(int(v) for v in c) for c in centers_rgb]
    return Image.fromarray(quantized_pixels), palette
