import os
import pathlib
import glob
import numpy as np
from PIL import Image
import multiprocessing
import tqdm 


# base_input_dir = "/media/ssd_backup/g2net/dataset/02_image_npy"
base_input_dir = "/home/yuri/kaggle/g2net-gravitational-wave-detection/dataset/02_image_npy"
base_output_dir = "/home/yuri/kaggle/g2net-gravitational-wave-detection/dataset/03_image_jpeg"


def arr2jpeg(f):
    fname = os.path.splitext(os.path.basename(f))[0] + ".jpeg"
    save_path = os.path.join(base_output_dir,train_test, fname)
    if os.path.exists(save_path):
        return 
    # print(f)
    # print(save_path)
    try:
        arr = np.load(f)
        img = Image.fromarray(arr)
        img.save(save_path)
    except:
        print("ERROR raise f={}".format(f))


def main(train_test):
    paths = glob.glob("{}/{}/*.npy".format(base_input_dir, train_test))

    # pool_obj.map(arr2jpeg, paths)
    for path in tqdm.tqdm(paths):
        # print(path)
        arr2jpeg(path)


for train_test in ["train", "test"]:
    # pool_obj = multiprocessing.Pool(6)
    main(train_test)
