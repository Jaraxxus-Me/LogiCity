import cv2
import os
import numpy as np

original_dir = "external/picked_transparent"
train_output_dir = "external/imgs/train"
test_output_dir = "external/imgs/test"

train_icon_ratio = 0.8

for type_name in os.listdir(original_dir):
    if type_name.startswith("."):
        continue
    type_dir = os.path.join(original_dir, type_name)
    type_dir_train = os.path.join(train_output_dir, type_name)
    type_dir_test = os.path.join(test_output_dir, type_name)
    print(type_dir)
    if not os.path.exists(type_dir_train):
        os.makedirs(type_dir_train)
    if not os.path.exists(type_dir_test):
        os.makedirs(type_dir_test)
    img_name_list = os.listdir(type_dir)
    train_len = int(len(img_name_list) * train_icon_ratio)
    train_list = img_name_list[:train_len]
    test_list = img_name_list[train_len:]
    # training icons
    idx = 0
    for img_name in train_list:
        if "png" in img_name:
            im = cv2.imread(os.path.join(type_dir, img_name))
            cv2.imwrite(os.path.join(type_dir_train, f"image_{idx}.png"), im)
            idx += 1
    # testing icons
    idx = 0
    for img_name in test_list:
        if "png" in img_name:
            im = cv2.imread(os.path.join(type_dir, img_name), cv2.IMREAD_UNCHANGED)
            cv2.imwrite(os.path.join(type_dir_test, f"image_{idx}.png"), im)
            idx += 1            
