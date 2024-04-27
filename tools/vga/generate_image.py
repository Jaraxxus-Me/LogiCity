from diffusers import StableDiffusionXLPipeline, DiffusionPipeline, AutoPipelineForImage2Image
import torch
from free_lunch_utils import register_free_upblock2d, register_free_crossattn_upblock2d
from diffusers.utils import load_image, make_image_grid
import ast
import os

base = AutoPipelineForImage2Image.from_pretrained(
    "/shared_data/p_vidalr/jinqiluo/model/diffuser/stable-diffusion-xl-base-1.0",  # this path should be "Your local dir" mentioned in readme.md
    torch_dtype=torch.float16, 
    variant="fp16", 
    use_safetensors=True,
    add_watermarker=False
).to("cuda")

refiner = AutoPipelineForImage2Image.from_pretrained(
    "/shared_data/p_vidalr/jinqiluo/model/diffuser/stable-diffusion-xl-refiner-1.0", # this path should be "Your local dir" mentioned in readme.md
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
    add_watermarker=False
).to("cuda")

register_free_upblock2d(base, b1=1.1, b2=1.2, s1=0.6, s2=0.4)
register_free_crossattn_upblock2d(base, b1=1.1, b2=1.2, s1=0.6, s2=0.4)
# base.enable_sequential_cpu_offload()

num_of_denoising_steps = 100
edit_strength_list = [0.8, 0.84, 0.88, 0.92, 0.96, 1.0]

####################################################

concept_file_path = "./llm_instruction/original_concept.txt"
image_file_path = "./llm_instruction/original_file_name.txt"

with open(concept_file_path, 'r') as file:
    content = file.read()
    concept_list = ast.literal_eval(content)[3:5] # [Ambulance car, Police car]

with open(image_file_path, 'r') as file:
    content = file.read()
    image_list = ast.literal_eval(content)[3:5] # [car_ambulance, car_police]

print("concept_list: ", concept_list)
print("image_list: ", image_list)

for each_concept, each_image in zip(concept_list, image_list):

    image_addr = f"../../imgs/{each_image}.png"
    image_init = load_image(image_addr)
    image_init_height, image_init_width = image_init.size
    image_gen_height, image_gen_width = int(768 * image_init_height / image_init_width), 768
    image_init = image_init.resize((image_gen_height, image_gen_width))

    description_path = f"./description/{each_image}.txt"
    with open(description_path, 'r') as file:
        content = file.read()
        prompt_list = ast.literal_eval(content)

    for each_index, each_prompt in enumerate(prompt_list):
        print(each_prompt)
        for edit_strength in edit_strength_list:
            for i in range(5):
                print(f"Generating image {i}")
                image_base = base(
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
                image_final = refiner(
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
                img_sub_dir_path = f"./variant_icons/{each_image}/rate_{edit_strength}"
                if not os.path.exists(img_sub_dir_path):
                    os.makedirs(img_sub_dir_path)
                image_final[0].save(f"{img_sub_dir_path}/{each_image}_prompt{each_index}_rate{edit_strength}_sample{i}.png")


        