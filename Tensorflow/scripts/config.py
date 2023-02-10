import os
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
from globals import PATHS, FILES, LABELS, PRETRAINED_MODEL_NAME
import tensorflow as tf

if __name__ == '__main__':
    config = config_util.get_configs_from_pipeline_file(FILES['PIPELINE_CONFIG'])

    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(FILES['PIPELINE_CONFIG'], "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline_config)

    pipeline_config.model.ssd.num_classes = len(LABELS)
    pipeline_config.train_config.batch_size = 1

    pipeline_config.train_config.fine_tune_checkpoint = os.path.join(PATHS['PRETRAINED_MODEL_PATH'],
                                                                     PRETRAINED_MODEL_NAME, 'checkpoint', 'ckpt-0')
    pipeline_config.train_config.fine_tune_checkpoint_type = "detection"

    # Training Setup
    pipeline_config.train_input_reader.label_map_path = FILES['LABELMAP']
    pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [os.path.join(PATHS['ANNOTATION_PATH'], 'train.record')]

    # Evaluation Setup
    pipeline_config.eval_input_reader[0].label_map_path = FILES['LABELMAP']
    pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [os.path.join(PATHS['ANNOTATION_PATH'], 'test.record')]

    config_text = text_format.MessageToString(pipeline_config)
    with tf.io.gfile.GFile(FILES['PIPELINE_CONFIG'], "wb") as f:
        f.write(config_text)
