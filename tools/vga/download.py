# Assuming you want to first download and save to a local path
cache_dir_base = "./external/cache_dir/base"
local_dir_base = "./external/local_dir/base"

cache_dir_refiner = "./external/cache_dir/refiner"
local_dir_refiner = "./external/local_dir/refiner"

from huggingface_hub import snapshot_download
snapshot_download(repo_id='stabilityai/stable-diffusion-xl-base-1.0', local_dir_use_symlinks=False, cache_dir=cache_dir_base, local_dir=local_dir_base)
snapshot_download(repo_id='stabilityai/stable-diffusion-xl-refiner-1.0', local_dir_use_symlinks=False, cache_dir=cache_dir_refiner, local_dir=local_dir_refiner)