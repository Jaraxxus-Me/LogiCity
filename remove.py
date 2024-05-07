# recursively remove .DS_store
import os
import sys


def remove_ds_store(path):
    for root, dirs, files in os.walk(path):
        if '.DS_Store' in files:
            os.remove(os.path.join(root, '.DS_Store'))

main_path = "imgs/all"
remove_ds_store(main_path)