from dataclasses import dataclass, field


@dataclass
class PipelineConfig:
    prompt: str
    output_dir: str = "output"
    model_id: str = "CompVis/stable-diffusion-v1-4"
    model_subfolder: str = ""
    num_colors: int = 16
    resolution: int = 64
    sd_steps: int = 30
    sd_guidance: float = 10.0
    negative_prompt: str = ""
    quantize_model: bool = False
    cpu_offload: bool = False
