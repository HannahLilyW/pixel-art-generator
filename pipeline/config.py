from dataclasses import dataclass, field


@dataclass
class PipelineConfig:
    prompt: str
    output_dir: str = "output"
    model_id: str = "CompVis/stable-diffusion-v1-4"
    model_subfolder: str = ""
    num_colors: int = 32
    resolution: int = 200
    sd_steps: int = 30
    sd_guidance: float = 10.0
    negative_prompt: str = ""
    cpu_offload: bool = False
