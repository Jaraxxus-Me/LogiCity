from diffusers import StableDiffusionXLPipeline, DiffusionPipeline, AutoPipelineForImage2Image
import torch
from free_lunch_utils import register_free_upblock2d, register_free_crossattn_upblock2d
from diffusers.utils import load_image, make_image_grid
import ast

base = AutoPipelineForImage2Image.from_pretrained(
    ".external/stable-diffusion-xl-base-1.0", 
    torch_dtype=torch.float16, 
    variant="fp16", 
    use_safetensors=True,
    add_watermarker=False
).to("cuda")

refiner = AutoPipelineForImage2Image.from_pretrained(
    "/shared_data/p_vidalr/jinqiluo/model/diffuser/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
    add_watermarker=False
).to("cuda")

register_free_upblock2d(base, b1=1.1, b2=1.2, s1=0.6, s2=0.4)
register_free_crossattn_upblock2d(base, b1=1.1, b2=1.2, s1=0.6, s2=0.4)

num_of_denoising_steps = 100
edit_strength = 0.75

####################################################

concept_file_path = "./llm_instruction/original_concept.txt"
image_file_path = "./llm_instruction/original_file_name.txt"

with open(concept_file_path, 'r') as file:
    content = file.read()
    concept_list = ast.literal_eval(content)

with open(image_file_path, 'r') as file:
    content = file.read()
    image_list = ast.literal_eval(content)

print("concept_list: ", concept_list)
print("image_list: ", image_list)

each_image = 'house1'
c=each_prompt = 'A futuristic glass house with a minimalist design, aerial view, outside view, pure background, single-instance, for tile-based games'

image_addr = f"./imgs/{each_image}.png"
image_init = load_image(image_addr)
image_init_height, image_init_width = image_init.size
image_gen_height, image_gen_width = int(768 * image_init_height / image_init_width), 768
image_init = image_init.resize((image_gen_height, image_gen_width))

description_path = f"./description/{each_image}.txt"
with open(description_path, 'r') as file:
    content = file.read()
    prompt_list = ast.literal_eval(content)



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
    image_final[0].save(f"./trytry/{each_image}_sample{i}.png")


        