import os
from globals import PATHS, FILES


if __name__ == '__main__':
    training_script = os.path.join(PATHS['API_MODEL'], 'research', 'object_detection', 'model_main_tf2.py')

    command = f'python {training_script} ' + \
              '--model_dir={} '.format(PATHS['CHECKPOINT_PATH']) + \
              '--pipeline_config_path={} '.format(FILES['PIPELINE_CONFIG']) + \
              '--num_train_steps=200'

    os.system(command)
