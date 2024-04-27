# Visual Generative Augmentation for LogiCity

## Installation
Actaully the libraries required are only diffuser and transformers and torch and openai. You can install them by running the following command:
```
pip install -r requirements.txt
```
Remember to download the whole pretrained model folder (SDXL base abd SDXL refiner) from the following link:
```
python3 tools/vga/download.py
```
Some pieces of huggingface repo download codes for your reference:
```
from huggingface_hub import snapshot_download
snapshot_download(repo_id='stabilityai/stable-diffusion-xl-base-1.0', local_dir_use_symlinks=False, cache_dir='Your local cache dir', local_dir='Your local dir')
snapshot_download(repo_id='stabilityai/stable-diffusion-xl-refiner-1.0', local_dir_use_symlinks=False, cache_dir='Your local cache dir', local_dir='Your local dir')
```
Then in the step 2-4 of the Usage, remember to change the model path to the path of the downloaded model folder in the **generate_image_xxx.py** and **multimodalgenerator.py**.

## Usage
The in-context instructions to the GPT, the index of the original images, and the corresponding concepts are in the folder **llm_instruction**. The original images are in the folder **imgs**.

1. To use GPT to generate various descriptions/prompts for a given concept, simply run the following command:
```
python generate_style.py
```
Note that you have to input your OpenAI API key in the file **generate_style.py**. The generated descriptions will be saved in the folder **description**.

2. To generate images based on the generated descriptions, run the following command:
```
python generate_image.py
```
The generated images will be saved in the folder **imgs_gen**.

3. If you want to simply generate some images based on one prompt you write yourself, you can run the following command:
```
generate_image_single_instance.py
```
The generated images will be saved in the folder **trytry**.

4. To use multiple GPUs for faster generation, you can run the following command with the slurm environment installed:
```
sbatch generate_image_distributed.sh
```
The generated images will be saved in the folder **imgs_gen_dist**.


