import cv2
import os
import numpy as np

original_dir_1 = "external/picked_transparent_chosen_from_SAM"
original_dir_2 = "external/picked_transparent_heuristic"
output_dir = "external/picked_transparent_merged"

# mode 0: choose from the two
# mode 1: merge the two
mode = 1

for type_name in os.listdir(original_dir_1):
    if type_name.startswith("."):
        continue
    type_dir_1 = os.path.join(original_dir_1, type_name)
    type_dir_2 = os.path.join(original_dir_2, type_name)
    print(type_dir_1, type_dir_2)
    type_dir_new = os.path.join(output_dir, type_name)
    if not os.path.exists(type_dir_new):
        os.makedirs(type_dir_new)

    for img_name in os.listdir(type_dir_1):
        im_1 = cv2.imread(os.path.join(type_dir_1, img_name), cv2.IMREAD_UNCHANGED)
        im_2 = cv2.imread(os.path.join(type_dir_2, img_name), cv2.IMREAD_UNCHANGED)
        height, width, channels = im_1.shape

        # print(type_name, img_name)
        # print("im_1: {}, im_2: {}".format(np.sum(im_1[:, :, 3])//255, np.sum(im_2[:, :, 3])//255))

        if mode == 0:
            # the more non-transparent pixels, the better
            if np.sum(im_1[:, :, 3])//255 > np.sum(im_2[:, :, 3])//255:
                # print("im_1")
                cv2.imwrite(os.path.join(type_dir_new, img_name), im_1)
            else:
                # print("im_2")
                cv2.imwrite(os.path.join(type_dir_new, img_name), im_2)
        elif mode == 1:
            # merge two pics
            new_im = np.ones((height, width, 4)) * 255
            new_im[:, :, :4] = im_1
            for i in range(height):
                for j in range(width):
                    if im_1[i, j, 3] == 0 and im_2[i, j, 3] != 0:
                        new_im[i, j] = im_2[i, j]
            cv2.imwrite(os.path.join(type_dir_new, img_name), new_im)