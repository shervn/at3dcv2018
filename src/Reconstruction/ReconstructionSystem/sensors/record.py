import sys, os
sys.path.append(".")
import json
from realsense_recorder import realsense_recorder

rel_path = os.path.realpath('')

# if __name__ == "__main__":
def record():

    config = rel_path + '/src/Reconstruction/ReconstructionSystem/config/realsense.json'
    if config is not None:
        with open(config) as json_file:
            config = json.load(json_file)

    output_folder = config['path_dataset']

    realsense_recorder(output_folder)
