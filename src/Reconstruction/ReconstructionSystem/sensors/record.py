import sys, os
sys.path.append(".")
import json
from realsense_recorder import realsense_recorder
from paths import record_config

rel_path = os.path.realpath('')

# if __name__ == "__main__":
def record():

    config = record_config
    if config is not None:
        with open(config) as json_file:
            config = json.load(json_file)

    output_folder = config['path_dataset']

    realsense_recorder(output_folder)
