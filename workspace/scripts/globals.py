import os

PRETRAINED_MODEL_NAME = 'ssd_resnet50_v1_fpn_640x640_coco17_tpu-8'
LABEL_MAP_NAME = 'label_map.pbtxt'
CUSTOM_MODEL_NAME = 'my_ssd_resnet50'

PATHS = {
    'ANNOTATION_PATH': os.path.join('..', 'annotations'),
    'API_MODEL': os.path.join('..', '..', 'models'),
    'DATASET_PATH': os.path.join('..', 'datasets'),
    'MODEL_PATH': os.path.join('..', 'models'),
    'PRETRAINED_MODEL_PATH': os.path.join('..', 'pre_trained_models'),
    'CHECKPOINT_PATH': os.path.join('..', 'models', CUSTOM_MODEL_NAME)
}

FILES = {
    'LABELMAP': os.path.join(PATHS['ANNOTATION_PATH'], LABEL_MAP_NAME),
    'PIPELINE_CONFIG': os.path.join('..', 'models', CUSTOM_MODEL_NAME, 'pipeline.config')
}

LABELS = ['stop', 'stopLeft',
          'warning', 'warningLeft',
          'go', 'goLeft',
          'pedestrianCrossing']
