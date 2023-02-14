import argparse
import os.path
from object_detection.utils import dataset_util
from lisa_loader import LisaLoader
import tensorflow as tf
from globals import LABELS, PATHS

DATASET_NAMES = ['LISA']

DATASET_PATHS = {
    'LISA': os.path.join('..', '..', 'dataset')
}

HEIGHT, WIDTH, IMAGE_FORMAT = 960, 1080, b'jpeg'
LABELS = tf.constant(LABELS)


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_filename', help='output filename', type=str)
    parser.add_argument('--test', help='load dataset for evaluating', action='store_true')

    return parser.parse_args()


def get_loader(dataset_name):
    if dataset_name == 'LISA':
        return LisaLoader()


def get_examples_info(split):
    examples = []

    for dataset_name in DATASET_NAMES:
        loader = get_loader(dataset_name)
        examples.extend(loader.get_examples(path_dataset=DATASET_PATHS[dataset_name], split=split))

    return examples


def split_examples(examples):
    labels, bb_cords = [], []

    img_paths = [example[0] for example in examples]
    boxes = [example[1] for example in examples]

    for boxes_img in boxes:
        labels.append([box[0] for box in boxes_img])
        bb_cords.append([box[1:] for box in boxes_img])

    return img_paths, labels, bb_cords


def serialize_example(filename, x_min, y_min, x_max, y_max, classes, classes_text):
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(HEIGHT),
        'image/width': dataset_util.int64_feature(WIDTH),
        'image/filename': dataset_util.bytes_feature(filename.numpy()),
        'image/source_id': dataset_util.bytes_feature(filename.numpy()),
        'image/encoded': dataset_util.bytes_feature(open(filename.numpy().decode('utf-8'), 'rb').read()),
        'image/format': dataset_util.bytes_feature(IMAGE_FORMAT),
        'image/object/bbox/xmin': dataset_util.float_list_feature(x_min.numpy()),
        'image/object/bbox/xmax': dataset_util.float_list_feature(x_max.numpy()),
        'image/object/bbox/ymin': dataset_util.float_list_feature(y_min.numpy()),
        'image/object/bbox/ymax': dataset_util.float_list_feature(y_max.numpy()),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text.numpy()),
        'image/object/class/label': dataset_util.int64_list_feature(classes.numpy()),
    }))
    return tf_example.SerializeToString()


def preprocess_example(filename, labels, bb_cord):
    # bounding boxes of one image
    bb_cord = tf.reshape(bb_cord, shape=tf.cast([-1, 4], dtype=tf.int32))

    x_min, y_min, x_max, y_max = tf.split(bb_cord, num_or_size_splits=4, axis=-1)

    # normalize the coordinates of bounding boxes wrt image's resolution
    x_min, y_min = tf.reshape(x_min / WIDTH, shape=[-1]), tf.reshape(y_min / HEIGHT, shape=[-1])
    x_max, y_max = tf.reshape(x_max / WIDTH, shape=[-1]), tf.reshape(y_max / HEIGHT, shape=[-1])

    labels = tf.reshape(labels, shape=[-1, 1])
    labels = tf.broadcast_to(labels, shape=[tf.shape(labels)[0], tf.shape(LABELS)[0]])

    # get the class' id for each bounding box
    classes = tf.reshape(tf.argmax(LABELS == labels, axis=-1) + 1, shape=[-1])
    classes_text = labels[:, 0]

    return filename, x_min, y_min, x_max, y_max, classes, classes_text


def tf_serialize_example(filename, x_min, y_min, x_max, y_max, classes, classes_text):
    # Serialize example
    tf_string = tf.py_function(
        serialize_example,
        (filename, x_min, y_min, x_max, y_max, classes, classes_text),
        tf.string)

    return tf.reshape(tf_string, ())


if __name__ == '__main__':
    args = read_args()
    writer = tf.io.TFRecordWriter(os.path.join(PATHS['ANNOTATION_PATH'], args.output_filename + '.record'))

    examples = get_examples_info(split=('test' if args.test else 'train'))

    img_paths, labels, bb_cords = split_examples(examples)

    dataset = tf.data.Dataset.from_tensor_slices((tf.ragged.constant(img_paths),
                                                  tf.ragged.constant(labels),
                                                  tf.ragged.constant(bb_cords)))
    dataset = dataset \
        .map(preprocess_example) \
        .map(tf_serialize_example) \
        .shuffle(5000)

    for example in dataset:
        writer.write(example.numpy())

    writer.close()
