import os

rel_path = os.path.realpath('') + '/'

sensors = rel_path + 'Reconstruction/ReconstructionSystem/sensors'
reconstruction_system = rel_path + 'Reconstruction/ReconstructionSystem/'
reconstruction_config = rel_path + 'Reconstruction/ReconstructionSystem/config/realsense.json'
utility = rel_path + 'Reconstruction/Utility'
record_config = rel_path + 'Reconstruction/ReconstructionSystem/config/realsense.json'
augmentation_util = rel_path + 'Augmentation/'
macpyrealsense2 = os.path.realpath('') + '/pyrealsenseformac/'
camera_config_path = rel_path + 'Reconstruction/ReconstructionSystem/config/realsense.json'
