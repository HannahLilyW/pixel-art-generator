# Pixel Art Generator

Generates clean, grid-aligned pixel art from a text prompt via a four-step pipeline:

1. **Generate** — Stable Diffusion (or whatever image model you want) produces a flat, cartoon-style image
2. **Quantize** — Reduce to a limited color palette
3. **Vectorize** — Convert to SVG for clean, sharp geometry
4. **Pixelate** — Render SVG at your chosen resolution with pixel-snapped colors

This method avoids the grid-alignment and inconsistent-pixel-size problems common in directly AI-generated "pixel art."

This project can also be used as a image-to-pixel-art converter if you bring your own image and skip the AI image generation part.

---

## Setup

**Requirements:** Python 3.12+, Homebrew

```bash
# System dependencies (one-time)
brew install python@3.12 cairo

# Create and activate virtual environment
python3.12 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

```bash
source venv/bin/activate
python3.12 cli.py "a cute frog wearing a hat"
```

The output lands in `./output/` — a `.png` pixel art file and a `.svg` vector file, both named with a timestamp.

### Options

| Flag | Default | Description |
|---|---|---|
| `--resolution`, `-r` | `64` | Output size in pixels (square) |
| `--colors`, `-c` | `16` | Number of colors in the palette |
| `--output`, `-o` | `output` | Output directory |
| `--input`, `-i` | — | Skip SD: use this image file instead |
| `--model` | `CompVis/stable-diffusion-v1-4` | HuggingFace model ID |
| `--steps` | `30` | SD inference steps (more = slower but better) |
| `--guidance` | `10.0` | SD guidance scale |
| `--save-intermediate` | off | Also save the raw SD and quantized images |

### Examples

```bash
# 32x32 sprite with 8 colors
python3.12 cli.py "a knight with a sword" -r 32 -c 8

# 128x128 with all intermediate steps saved
python3.12 cli.py "a fantasy castle at sunset" -r 128 --save-intermediate

# Skip SD — vectorize and pixelate an existing image
python3.12 cli.py "my image" --input my_drawing.png -r 64 -c 16

# Use SDXL for higher quality (needs ~8 GB VRAM / RAM)
python3.12 cli.py "a space rocket" --model stabilityai/stable-diffusion-xl-base-1.0
```

---

## Model notes

The default model (`CompVis/stable-diffusion-v1-4`) requires no HuggingFace login. For other options:

- **Higher quality, requires HF login**: `stabilityai/stable-diffusion-xl-base-1.0`
- **Fine-tuned for anime style, heavier model:** `SeeSee21/Z-Anime`

The pipeline automatically appends flat-art style keywords to your prompt, so you don't need to add them yourself.

---

## Output files

```
output/
  a-cute-frog-wearing-a-hat_20240101_120000_pixel_64x64.png   ← pixel art
  a-cute-frog-wearing-a-hat_20240101_120000_vector.svg        ← editable vector
  a-cute-frog-wearing-a-hat_20240101_120000_sd.png            ← (with --save-intermediate)
  a-cute-frog-wearing-a-hat_20240101_120000_quantized.png     ← (with --save-intermediate)
```

The SVG is the editable intermediate — tweak shapes and colors there, then re-run the pixelate step to update the pixel art.

Or edit the generated pixel art directly.

---

## Hardware

- **Apple Silicon (M1/M2/M3):** uses MPS automatically
- **NVIDIA GPU:** uses CUDA automatically
- **CPU fallback:** works but generation takes several minutes
