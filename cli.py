import os
import re
import sys
from datetime import datetime
from pathlib import Path

import click
from PIL import Image

from pipeline import PipelineConfig, run_pipeline


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower())[:40].strip("-")


@click.command()
@click.argument("prompt")
@click.option("--resolution", "-r", default=64, show_default=True, help="Output pixel art size (square).")
@click.option("--colors", "-c", default=16, show_default=True, help="Number of colors in the palette.")
@click.option("--output", "-o", default="output", show_default=True, help="Output directory.")
@click.option("--input", "-i", "input_path", default=None, help="Skip SD: use this image instead.")
@click.option("--model", default=None, help="HuggingFace model ID (default: CompVis/stable-diffusion-v1-4).")
@click.option("--subfolder", default=None, help="Subfolder within the HuggingFace repo containing the diffusers config (e.g. 'diffusers').")
@click.option("--steps", default=30, show_default=True, help="SD inference steps.")
@click.option("--guidance", default=10.0, show_default=True, help="SD guidance scale.")
@click.option("--save-intermediate", is_flag=True, help="Save SD and quantized images alongside final output.")
@click.option("--quantize", is_flag=True, help="Quantize transformer/UNet weights to FP8 via optimum-quanto before inference (reduces VRAM).")
@click.option("--offload", is_flag=True, help="Sequential CPU offload: move each model to MPS only while active, freeing MPS budget between stages.")
@click.option("--negative-prompt", "negative_prompt", default=None, help="Negative prompt string.")
def generate(prompt, resolution, colors, output, input_path, model, subfolder, steps, guidance, save_intermediate, quantize, offload, negative_prompt):
    """Generate pixel art from a text PROMPT."""
    config_kwargs = dict(
        prompt=prompt,
        output_dir=output,
        num_colors=colors,
        resolution=resolution,
        sd_steps=steps,
        sd_guidance=guidance,
    )
    if model is not None:
        config_kwargs["model_id"] = model
    if subfolder is not None:
        config_kwargs["model_subfolder"] = subfolder
    if quantize:
        config_kwargs["quantize_model"] = True
    if offload:
        config_kwargs["cpu_offload"] = True
    if negative_prompt is not None:
        config_kwargs["negative_prompt"] = negative_prompt
    config = PipelineConfig(**config_kwargs)

    input_image = None
    if input_path:
        input_image = Image.open(input_path)
        print(f"Using input image: {input_path}")

    try:
        pixel_art, svg_content, intermediates = run_pipeline(config, input_image)
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)

    Path(output).mkdir(parents=True, exist_ok=True)
    slug = f"{_slugify(prompt)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    pixel_path = os.path.join(output, f"{slug}_pixel_{resolution}x{resolution}.png")
    svg_path = os.path.join(output, f"{slug}_vector.svg")

    pixel_art.save(pixel_path)
    with open(svg_path, "w") as f:
        f.write(svg_content)

    click.echo(f"Pixel art -> {pixel_path}")
    click.echo(f"Vector    -> {svg_path}")

    if save_intermediate:
        sd_path = os.path.join(output, f"{slug}_sd.png")
        q_path = os.path.join(output, f"{slug}_quantized.png")
        intermediates["sd_image"].save(sd_path)
        intermediates["quantized"].save(q_path)
        click.echo(f"SD image  -> {sd_path}")
        click.echo(f"Quantized -> {q_path}")


if __name__ == "__main__":
    generate()
