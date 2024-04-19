import os
import sys
import csv
from scipy.special import softmax
import argparse
import numpy as np
import multiprocessing
import time
import json
import time
from time import sleep

from typing import Dict, List, Callable
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import torch.distributed as dist
import torch.multiprocessing as mp

from free_lunch_utils import register_free_upblock2d, register_free_crossattn_upblock2d
from diffusers.utils import load_image, make_image_grid
from multimodalgenerator import MultimodalGenerator
import ast
import subprocess


def run_inference(rank, world_size, generator, concept_list, image_list):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    generator.base.to(rank)
    generator.refiner.to(rank)
    # generator.base.enable_sequential_cpu_offload()
    register_free_upblock2d(generator.base, b1=1.1, b2=1.2, s1=0.6, s2=0.4)
    register_free_crossattn_upblock2d(generator.base, b1=1.1, b2=1.2, s1=0.6, s2=0.4)
    num_of_denoising_steps = 50
    edit_strength = 1

    for each_concept, each_image in zip(concept_list, image_list):

        image_addr = f"./imgs/{each_image}.png"
        image_init = load_image(image_addr)
        image_init_height, image_init_width = image_init.size
        image_gen_height, image_gen_width = int(768 * image_init_height / image_init_width), 768
        image_init = image_init.resize((image_gen_height, image_gen_width))

        description_path = f"./description/{each_image}.txt"
        with open(description_path, 'r') as file:
            content = file.read()
            prompt_list = ast.literal_eval(content)

        for each_index, each_prompt in enumerate(prompt_list):

            image_base = generator.base(
                prompt=[each_prompt],
                negative_prompt=[''],
                image=image_init,
                num_inference_steps=num_of_denoising_steps,
                denoising_end=0.8,
                output_type="latent",
                strength=edit_strength,
                height=image_gen_height,
                width=image_gen_width,
            ).images
            image_final = generator.refiner(
                prompt=[each_prompt],
                negative_prompt=[''],
                image=image_base,
                num_inference_steps=num_of_denoising_steps,
                denoising_start=0.8,
                height=image_gen_height,
                width=image_gen_width,
            ).images

            # reshape the image to the original aspect ratio
            image_final[0] = image_final[0].resize((image_init_height, image_init_width))
            image_final[0].save(f"./imgs_gen_dist/{each_image}_prompt{each_index}_sample{torch.distributed.get_rank()}.png")

def get_master_address():
    # Get the full node list in a more usable format
    command = ['scontrol', 'show', 'hostnames', os.environ['SLURM_JOB_NODELIST']]
    result = subprocess.run(command, stdout=subprocess.PIPE)
    master_addr = result.stdout.decode('utf-8').strip().split('\n')[0]  # Takes the first hostname
    return master_addr

def main():

    generator = MultimodalGenerator()

    concept_file_path = "./llm_instruction/original_concept.txt"
    image_file_path = "./llm_instruction/original_file_name.txt"

    with open(concept_file_path, 'r') as file:
        content = file.read()
        concept_list = ast.literal_eval(content)[3:]

    with open(image_file_path, 'r') as file:
        content = file.read()
        image_list = ast.literal_eval(content)[3:]

    print("concept_list: ", concept_list)
    print("image_list: ", image_list)

    os.environ['MASTER_ADDR'] = get_master_address()
    os.environ['MASTER_PORT'] = '12357'  
    world_size = 5
   
    mp.spawn(run_inference, args=(world_size, generator, concept_list, image_list, ), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()