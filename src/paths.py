import os

rel_path = os.path.realpath('') + '/'
index = len('src/')
rel_path_before_src = os.path.realpath('')[0:-index]

sensors = rel_path + 'Reconstruction/ReconstructionSystem/sensors'
reconstruction_system = rel_path + 'Reconstruction/ReconstructionSystem/'
reconstruction_config = rel_path + 'Reconstruction/ReconstructionSystem/config/realsense.json'
utility = rel_path + 'Reconstruction/Utility'
record_config = rel_path + 'Reconstruction/ReconstructionSystem/config/realsense.json'
augmentation_util = rel_path + 'Augmentation/'
camera_config_path = rel_path + 'Reconstruction/ReconstructionSystem/config/realsense.json'


# reconstructed_scene = rel_path_before_src + '/Samples/scene0000_00_vh_clean_2.ply'
# segmented_reconstructed_scene = rel_path_before_src + '/Samples/scene0000_00_vh_clean_2.labels.ply'

reconstructed_scene = rel_path_before_src + '/Samples/final_integrated_ordered.ply'
segmented_reconstructed_scene = rel_path_before_src + '/Samples/final_labled_integrated_ordered.ply'


bed_pc_name = rel_path_before_src + '/Samples/new_bed_clean.ply'
render_option_path = rel_path_before_src + '/Data/renderoption.json'
camera_trajectory_path = rel_path_before_src + '/Data/camera_trajectory.json'

macpyrealsense2 = rel_path_before_src + '/pyrealsenseformac/'

furnitures_path = rel_path_before_src + '/Data/Furnitures/'
logo_path = rel_path_before_src + '/Images/logo.png'

layout_path = rel_path_before_src + '/Layout'
images_path = rel_path_before_src + '/Images'

