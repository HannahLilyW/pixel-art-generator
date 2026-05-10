import os
import re

import pytest
from PIL import Image

from pipeline import PipelineConfig, run_pipeline

OUTPUT_DIR = "test_output"


def _run(model_id, steps, guidance):
    config = PipelineConfig(
        prompt="a cute anime cat, flat cartoon style",
        model_id=model_id,
        num_colors=16,
        resolution=200,
        sd_steps=steps,
        sd_guidance=guidance,
    )
    pixel_art, svg_content, intermediates = run_pipeline(config)
    assert isinstance(pixel_art, Image.Image)
    assert isinstance(svg_content, str)
    assert pixel_art.size == (200, 200)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", model_id.lower())

    paths = {
        "pixel":     os.path.join(OUTPUT_DIR, f"{slug}_pixel.png"),
        "sd":        os.path.join(OUTPUT_DIR, f"{slug}_sd.png"),
        "quantized": os.path.join(OUTPUT_DIR, f"{slug}_quantized.png"),
        "vector":    os.path.join(OUTPUT_DIR, f"{slug}_vector.svg"),
    }

    pixel_art.save(paths["pixel"])
    intermediates["sd_image"].save(paths["sd"])
    intermediates["quantized"].save(paths["quantized"])
    with open(paths["vector"], "w") as f:
        f.write(svg_content)

    print()
    for label, path in paths.items():
        print(f"  {label}: {path}")


def test_pipeline_sd_v1_4():
    _run("CompVis/stable-diffusion-v1-4", steps=30, guidance=10.0)


@pytest.mark.requires_hf_auth
def test_pipeline_sdxl():
    _run("stabilityai/stable-diffusion-xl-base-1.0", steps=30, guidance=10.0)


@pytest.mark.requires_hf_auth
def test_pipeline_z_image_turbo():
    _run("Tongyi-MAI/Z-Image-Turbo", steps=9, guidance=1.0)
