import os

PRETRAINED_MODEL_NAME = 'ssd_resnet50_v1_fpn_640x640_coco17_tpu-8'
LABEL_MAP_NAME = 'label_map.pbtxt'
CUSTOM_MODEL_NAME = 'my_ssd_resnet'

PATHS = {
    'ANNOTATION_PATH': os.path.join('..', 'workspace', 'annotations'),
    'DATASET_PATH': os.path.join('../../dataset'),
    'MODEL_PATH': os.path.join('..', 'workspace', 'models'),
    'PRETRAINED_MODEL_PATH': os.path.join('..', 'workspace', 'pre_trained_model'),
    'CHECKPOINT_PATH': os.path.join('..', 'workspace', 'models', CUSTOM_MODEL_NAME),
    'OUTPUT_PATH': os.path.join('..', 'workspace', 'models', CUSTOM_MODEL_NAME, 'export')
}

FILES = {
    'LABELMAP': os.path.join(PATHS['ANNOTATION_PATH'], LABEL_MAP_NAME),
    'PIPELINE_CONFIG': os.path.join('..', 'workspace', 'models', CUSTOM_MODEL_NAME, 'pipeline.config')
}

LABELS = ['stop', 'stopLeft',
          'warning', 'warningLeft',
          'go', 'goLeft',
          'pedestrianCrossing']
