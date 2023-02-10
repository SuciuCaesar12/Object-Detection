import argparse
import os
from globals import PATHS, FILES


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_model_path', type=str)

    return parser.parse_args()


if __name__ == '__main__':
    args = read_args()

    training_script = os.path.join(args.api_model_path, 'research', 'object_detection', 'model_main_tf2.py')

    command = f'python {training_script} ' + \
              '--model_dir={} '.format(PATHS['CHECKPOINT_PATH']) + \
              '--pipeline_config_path={} '.format(FILES['PIPELINE_CONFIG']) + \
              '--num_train_steps=1000'

    os.system(command)
