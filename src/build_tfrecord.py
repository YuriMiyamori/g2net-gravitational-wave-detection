import tensorflow as tf
import glob
import random
import os
import pandas as pd
from tqdm import tqdm
import shutil
num_tfrecords = -1

output_path = "/home/yuri/kaggle/g2net-gravitational-wave-detection/dataset/04_tfrecord"
input_path = "/home/yuri/kaggle/g2net-gravitational-wave-detection/dataset/03_image_jpeg"
num_samples = 2048

train_label_df = pd.read_csv("/home/yuri/kaggle/g2net-gravitational-wave-detection/training_labels.csv")
label_dict = train_label_df.set_index("id").T.to_dict('list')

def set_label(name):
    return label_dict[name][0]

import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]

# 下記の関数を使うと値を tf.Example と互換性の有る型に変換できる

def _bytes_feature(value):
  """string / byte 型から byte_list を返す"""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """float / double 型から float_list を返す"""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """bool / enum / int / uint 型から Int64_list を返す"""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# 関連する特徴量のディクショナリを作成
def train_image_example(path):
    image_string = open(path, 'rb').read()
    id_string  = os.path.splitext(os.path.basename(path))[0].split("_")[0]
    label = set_label(id_string)
    # print(id_string)
    # print(label)

    feature = {
      'label': _int64_feature(label),
      'image_raw': _bytes_feature(image_string),
      'id': _bytes_feature(id_string.encode('utf-8')),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def test_image_example(path):
    image_string = open(path, 'rb').read()
    id_string  = os.path.splitext(os.path.basename(path))[0].split("_")[0]

    feature = {
      'image_raw': _bytes_feature(image_string),
      'id': _bytes_feature(id_string.encode('utf-8')),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))
  

def write_tfrecord(paths, tfrec_num):
    out_path = os.path.join(output_path, train_test, "record_{}-{}.tfrec".format(tfrec_num, num_tfrecords-1))
    with tf.io.TFRecordWriter(out_path) as writer:
        for path in paths:
            # image_path = f"{images_dir}/{sample['image_id']:012d}.jpg"
            if train_test == "train":
                example = train_image_example(path)
            else:
                example = test_image_example(path)
            writer.write(example.SerializeToString())


def main(train_test):
    paths = glob.glob(os.path.join(input_path, train_test) + "/*.jpeg")
    global num_tfrecords
    num_tfrecords = len(paths) // num_samples
    if len(paths) % num_samples:
        num_tfrecords += 1  # add one record if there are any remaining samples
    if train_test == "train":
        random.shuffle(paths)
    else:
        paths = sorted(paths)
        # paths = sorted(paths, key=natural_keys)

    print(paths[:10])
    for p in paths[:10]:
         print(os.path.splitext(os.path.basename(p))[0])
         
    for tfrec_num in tqdm(range(num_tfrecords)):
        paths_chunked = paths[(tfrec_num * num_samples) : ((tfrec_num + 1) * num_samples)]
        write_tfrecord(paths_chunked, tfrec_num)


for train_test in ["test"]:
# for train_test in ["train", "test"]:
    p = os.path.join(output_path, train_test)
    shutil.rmtree(p)
    os.makedirs(p)
    main(train_test)
