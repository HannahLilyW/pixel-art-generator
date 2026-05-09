from PIL import Image

from .config import PipelineConfig

# Appended to every prompt to steer SD toward flat, vectorizable output
_STYLE_SUFFIX = (
    "cartoon, flat"
    # "flat style, cel shaded, cartoon style, limited color palette, no outlines, solid colors"
    # "bold outlines, clean vector art style, simple shapes, solid colors"
)


def _load_pipeline(model_id, subfolder=None, **kwargs):
    from diffusers import AutoPipelineForText2Image

    def _try_load(path, **kw):
        try:
            return AutoPipelineForText2Image.from_pretrained(path, **kw)
        except TypeError:
            # safety_checker kwargs are SD 1.x-specific; drop them for other architectures
            for key in ("safety_checker", "requires_safety_checker"):
                kw.pop(key, None)
            return AutoPipelineForText2Image.from_pretrained(path, **kw)

    if subfolder:
        # AutoPipelineForText2Image doesn't forward subfolder to its initial load_config,
        # so snapshot_download the repo and point from_pretrained at the local subfolder.
        from huggingface_hub import snapshot_download
        import os
        try:
            print(f"Loading model '{model_id}/{subfolder}' from local cache...")
            local_dir = snapshot_download(model_id, local_files_only=True)
        except Exception:
            print(f"Not cached — downloading '{model_id}/{subfolder}' from HuggingFace...")
            local_dir = snapshot_download(model_id, allow_patterns=[f"{subfolder}/**"])
        return _try_load(os.path.join(local_dir, subfolder), **kwargs)
    else:
        try:
            print(f"Loading model '{model_id}' from local cache...")
            return _try_load(model_id, local_files_only=True, **kwargs)
        except OSError:
            print(f"Not cached — downloading '{model_id}' from HuggingFace...")
            return _try_load(model_id, **kwargs)


# These models produce NaN on MPS with bfloat16/float16 and require float32
_FLOAT32_MPS_MODELS = {
    "CompVis/stable-diffusion-v1-4",
    "runwayml/stable-diffusion-v1-5",
    "stabilityai/stable-diffusion-xl-base-1.0",
}


def generate_image(config: PipelineConfig) -> Image.Image:
    import torch

    needs_float32_mps = config.model_id in _FLOAT32_MPS_MODELS
    if torch.backends.mps.is_available():
        device, dtype = "mps", torch.float32 if needs_float32_mps else torch.bfloat16
    elif torch.cuda.is_available():
        device, dtype = "cuda", torch.float16
    else:
        device, dtype = "cpu", torch.float32

    kwargs = dict(torch_dtype=dtype, safety_checker=None, requires_safety_checker=False)

    pipe = _load_pipeline(config.model_id, subfolder=config.model_subfolder or None, **kwargs)

    pipe = pipe.to(device)

    pipe.enable_attention_slicing()

    full_prompt = f"{config.prompt}, {_STYLE_SUFFIX}"
    print(f"Prompt: {full_prompt}")

    gen_kwargs = dict(
        num_inference_steps=config.sd_steps,
        guidance_scale=config.sd_guidance,
    )
    if config.negative_prompt:
        gen_kwargs["negative_prompt"] = config.negative_prompt

    result = pipe(full_prompt, **gen_kwargs)

    return result.images[0]
