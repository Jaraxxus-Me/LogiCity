import os
import argparse
from glob import glob
from PIL import Image

def create_gif(image_folder, output_file):
    image_files = glob(os.path.join(image_folder, "*.png"))
    image_files.sort()
    n_images = len(image_files)

    # Open the first image
    with Image.open(image_files[0]) as im:
        # Save the first image as the animated GIF
        im.save(output_file, save_all=True,
                append_images=[Image.open(os.path.join(image_folder, "step_{:04}.png".format(k))) for k in range(1, n_images - 1)],
                duration=200, loop=0)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Create an animated GIF from a sequence of images.")
    parser.add_argument("image_folder", help="Path to the folder containing image files.")
    parser.add_argument("output_file", help="Path to the output GIF file.")
    
    args = parser.parse_args()

    # Call the function with provided arguments
    create_gif(args.image_folder, args.output_file)
