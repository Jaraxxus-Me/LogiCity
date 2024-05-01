import cv2
import os
import numpy as np

original_dir = "external/picked"
output_dir = "external/picked_transparent_heuristic"


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
            new_im = np.ones((height, width, 4)) * 255
            new_im[:, :, :3] = im
            background_color_1 = im[5, 5, :3].tolist()
            background_color_2 = im[-5, -5, :3].tolist()
            background_color_3 = im[1, 1, :3].tolist()
            background_color_4 = im[-2, -2, :3].tolist()
            for i in range(height):
                for j in range(width):
                    cur_color = new_im[i, j, :3].tolist()
                    # if cur_color == background_color:
                    if abs(cur_color[0]-background_color_1[0])+abs(cur_color[1]-background_color_1[1])+abs(cur_color[2]-background_color_1[2])<200 \
                    or abs(cur_color[0]-background_color_2[0])+abs(cur_color[1]-background_color_2[1])+abs(cur_color[2]-background_color_2[2])<200 \
                    or abs(cur_color[0]-background_color_3[0])+abs(cur_color[1]-background_color_3[1])+abs(cur_color[2]-background_color_3[2])<200 \
                    or abs(cur_color[0]-background_color_4[0])+abs(cur_color[1]-background_color_4[1])+abs(cur_color[2]-background_color_4[2])<200:
                        new_im[i, j, :] = np.array([255.0, 255.0, 255.0, 0])
            cv2.imwrite(os.path.join(type_dir_new, f"image_{idx}.png"), new_im)
            idx += 1
