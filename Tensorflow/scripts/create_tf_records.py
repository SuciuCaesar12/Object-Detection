import imghdr
import os.path
import tensorflow as tf
import tqdm
import argparse
import cv2
from object_detection.utils import dataset_util
from Tensorflow.scripts.lisa_loader import get_samples_info
from Tensorflow.scripts.globals import LABELS, PATHS


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--record_name', type=str)

    return parser.parse_args()


def create_tf_example(sample):
    filename, boxes = sample

    # with tf.io.gfile.GFile(filename, 'rb') as fid:
    #     encoded_image_data = fid.read()

    encoded_image_data = open(filename, 'rb').read()
    try:
        # height, width, _ = tf.io.decode_jpeg(filename).shape
        height, width, _ = cv2.imread(filename).shape
    except:
        print('File format not supported by Tensorflow: {}\n'.format(imghdr.what(filename)) +
              "Filename: {}".format(filename))

    filename = filename.encode('utf8')
    image_format = b'jpeg'

    xmins = []  # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = []  # List of normalized right x coordinates in bounding box
    # (1 per box)
    ymins = []  # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = []  # List of normalized bottom y coordinates in bounding box
    # (1 per box)
    classes_text = []  # List of string class name of bounding box (1 per box)
    classes = []  # List of integer class id of bounding box (1 per box)

    for box in boxes:
        label, x_min, y_min, x_max, y_max = box

        xmins.append(float(x_min / width))
        ymins.append(float(y_min / height))
        xmaxs.append(float(x_max / width))
        ymaxs.append(float(y_max / height))
        classes_text.append(label.encode('utf8'))
        classes.append(LABELS.index(label) + 1)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def create_tf_records():
    args = read_args()

    writer = tf.io.TFRecordWriter(os.path.join(PATHS['ANNOTATION_PATH'], args.record_name + '.record'))

    examples = get_samples_info(path_dataset=PATHS['DATASET_PATH'], split='test')

    for example in tqdm.tqdm(examples, disable=False):
        tf_example = create_tf_example(example)
        writer.write(tf_example.SerializeToString())

    writer.close()


if __name__ == '__main__':
    create_tf_records()
