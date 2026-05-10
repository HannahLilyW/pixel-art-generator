import glob
import os

import torch
from PIL import Image

from .config import PipelineConfig


def _apply_fp8_scales(text_encoder):
    """
    FP8-quantized text encoders store weights as float8_e4m3fn with sibling .scale tensors.
    Standard from_pretrained casts the FP8 values to BF16 but drops the scale tensors
    (logged as UNEXPECTED), leaving every linear layer off by its scale factor.
    Fix: load only the tiny scalar scales and multiply the already-loaded weights in-place.
    """
    from safetensors import safe_open

    model_path = text_encoder.config._name_or_path
    shards = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not shards:
        return

    scales = {}
    has_fp8 = False
    for shard in shards:
        with safe_open(shard, framework="pt") as st:
            for k in st.keys():
                t = st.get_tensor(k)
                if t.dtype == torch.float8_e4m3fn:
                    has_fp8 = True
                elif k.endswith(".scale"):
                    scales[k] = t

    if not has_fp8 or not scales:
        return

    print(f"Text encoder: applying {len(scales)} FP8 scale factors to correct BF16 weights...")
    with torch.no_grad():
        for scale_key, scale_val in scales.items():
            weight_key = scale_key[: -len(".scale")] + ".weight"
            parts = weight_key.split(".")
            try:
                module = text_encoder
                for part in parts[:-1]:
                    module = getattr(module, part)
                param = getattr(module, parts[-1])
                if isinstance(param, torch.nn.Parameter):
                    param.data.mul_(scale_val.to(param.dtype))
            except AttributeError:
                pass


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

    if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
        _apply_fp8_scales(pipe.text_encoder)

    if config.quantize_model:
        from optimum.quanto import freeze, quantize as quanto_quantize
        # FP8 matmul is only natively supported on CUDA; MPS and CPU must use INT8
        if device == "cuda":
            from optimum.quanto import qfloat8
            qtype, qtype_name = qfloat8, "FP8"
        else:
            from optimum.quanto import qint8
            qtype, qtype_name = qint8, "INT8"
        for attr in ("transformer", "unet"):
            target = getattr(pipe, attr, None)
            if target is not None and isinstance(target, torch.nn.Module):
                print(f"Quantizing {type(target).__name__} weights to {qtype_name} (optimum-quanto)...")
                quanto_quantize(target, weights=qtype)
                freeze(target)

    if config.cpu_offload:
        # enable_sequential_cpu_offload conflicts with quanto tensors (uses meta device internally);
        # enable_model_cpu_offload works at the component level and only calls .to(), which quanto supports.
        pipe.enable_model_cpu_offload(device=device)
    else:
        pipe = pipe.to(device)

    pipe.enable_attention_slicing()

    full_prompt = config.prompt
    print(f"Prompt: {full_prompt}")

    gen_kwargs = dict(
        num_inference_steps=config.sd_steps,
        guidance_scale=config.sd_guidance,
        height=512,
        width=512,
    )
    if config.negative_prompt:
        gen_kwargs["negative_prompt"] = config.negative_prompt

    result = pipe(full_prompt, **gen_kwargs)

    return result.images[0]
