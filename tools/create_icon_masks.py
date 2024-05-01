import cv2
import os
import numpy as np
from segment_anything import SamPredictor, sam_model_registry

original_dir = "external/picked"
output_dir = "external/picked_transparent_single_point"

sam = sam_model_registry["default"](checkpoint="/home/data2/qiweidu/logicity/sam_weights/sam_vit_h_4b8939.pth")
predictor = SamPredictor(sam)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for type_name in os.listdir(original_dir):
    if type_name.startswith("."):
        continue
    type_dir = os.path.join(original_dir, type_name)
    type_dir_new = os.path.join(output_dir, type_name)
    print(type_dir)
    if not os.path.exists(type_dir_new):
        os.makedirs(type_dir_new)
    idx = 0
    for img_name in os.listdir(type_dir):
        if "png" in img_name:
            im = cv2.imread(os.path.join(type_dir, img_name))
            height, width, channels = im.shape
            predictor.set_image(im)
            # SAM with multiple-point prompts
            # masks, _, _ = predictor.predict(
            #     point_coords=np.array([
            #         [width//2, height//2],
            #         [width//2, height//4],
            #         [width//2, height//8],
            #         [width//2, height*3//4],
            #         [width//2, height*7//8],
            #     ]), 
            #     point_labels=np.array([1, 1, 1, 1, 1]),
            #     multimask_output = False
            # )
            # SAM with single-point prompts
            masks, _, _ = predictor.predict(
                point_coords=np.array([
                    [width//2, height//2],
                ]), 
                point_labels=np.array([1]),
                multimask_output = True
            )
            mask_idx = masks.shape[0] - 1
            new_im = np.ones((height, width, 4)) * 255
            new_im[:, :, :3] = im
            for i in range(height):
                for j in range(width):
                    if masks[mask_idx][i, j] == False:
                        new_im[i, j, :] = np.array([255.0, 255.0, 255.0, 0])
            cv2.imwrite(os.path.join(type_dir_new, f"image_{idx}.png"), new_im)
            print(f"image_{idx}.png")
            idx += 1
