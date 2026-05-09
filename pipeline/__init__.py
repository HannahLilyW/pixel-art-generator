from __future__ import annotations

from PIL import Image

from .config import PipelineConfig
from .generate import generate_image
from .quantize import quantize_colors
from .vectorize import vectorize_image
from .pixelate import pixelate


def run_pipeline(
    config: PipelineConfig,
    input_image: Image.Image | None = None,
) -> tuple[Image.Image, str, dict]:
    """
    Run the full pixel-art pipeline.

    - input_image: if provided, skip the Stable Diffusion step.

    Returns (pixel_art, svg_content, intermediates) where intermediates
    contains 'sd_image' and 'quantized' PIL images for optional saving.
    """
    intermediates: dict = {}

    if input_image is None:
        sd_image = generate_image(config)
    else:
        sd_image = input_image.convert("RGB")
    intermediates["sd_image"] = sd_image

    print("Quantizing colors...")
    quantized, palette_colors = quantize_colors(sd_image, config.num_colors)
    intermediates["quantized"] = quantized

    print("Vectorizing...")
    svg_content = vectorize_image(quantized, config.num_colors)
    intermediates["svg"] = svg_content

    print(f"Rendering pixel art at {config.resolution}x{config.resolution}...")
    pixel_art = pixelate(
        svg_content,
        config.resolution,
        # palette_colors
    )

    return pixel_art, svg_content, intermediates
