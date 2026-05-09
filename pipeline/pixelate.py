from __future__ import annotations

import io

import numpy as np
from PIL import Image


def pixelate(svg_content: str, resolution: int) -> Image.Image:
    import cairosvg

    oversample = 8
    png_data = cairosvg.svg2png(
        bytestring=svg_content.encode(),
        output_width=resolution * oversample,
        output_height=resolution * oversample,
    )
    high_res = np.array(Image.open(io.BytesIO(png_data)).convert("RGBA"))

    H, W = high_res.shape[:2]
    out_h, out_w = H // oversample, W // oversample
    out = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    for row in range(out_h):
        for col in range(out_w):
            block = high_res[
                row * oversample : (row + 1) * oversample,
                col * oversample : (col + 1) * oversample,
            ]
            # Prefer fully-opaque pixels to ignore anti-aliased edges entirely
            opaque = block[block[:, :, 3] == 255, :3]
            pixels = opaque if len(opaque) else block[:, :, :3].reshape(-1, 3)
            unique_colors, counts = np.unique(pixels.reshape(-1, 3), axis=0, return_counts=True)
            out[row, col] = unique_colors[counts.argmax()]

    return Image.fromarray(out, "RGB")
