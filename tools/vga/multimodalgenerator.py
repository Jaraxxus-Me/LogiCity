from diffusers import StableDiffusionXLPipeline, DiffusionPipeline, AutoPipelineForImage2Image
import torch
import ast

import torch.distributed as dist
import torch.multiprocessing as mp

from free_lunch_utils import register_free_upblock2d, register_free_crossattn_upblock2d


class MultimodalGenerator():

    def __init__(self):

        self.base = AutoPipelineForImage2Image.from_pretrained(
            "/shared_data/p_vidalr/jinqiluo/model/diffuser/stable-diffusion-xl-base-1.0", 
            torch_dtype=torch.float16, 
            variant="fp16", 
            use_safetensors=True,
            add_watermarker=False
        )

        self.refiner = AutoPipelineForImage2Image.from_pretrained(
            "/shared_data/p_vidalr/jinqiluo/model/diffuser/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=self.base.text_encoder_2,
            vae=self.base.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            add_watermarker=False
        )




