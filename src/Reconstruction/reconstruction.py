from open3d import *

# pointcloud_name = '../Samples/scene0000_00_vh_clean_2.ply'

import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
print(os.listdir(dir_path))
sys.path.append("./ReconstructionSystem")
from run_system import run_system


class Reconstructor:

    def __init__(self):
        #only for testing
        # self.reconstructed_pointcloud = read_point_cloud(pointcloud_name)
        print("initialization")
        config = "ReconstructionSystem/config/realsense.json"
        make = True
        register = True
        refine = True
        integrate = True
        debug_mode = False

        run_system(config, make, register, refine, integrate, debug_mode)

    # if __name__ == "__main__":
    #     config = "Reconstruction System/config/realsense.json"
    #     make = True
    #     register = True
    #     refine = True
    #     integrate = True
    #     debug_mode = False
    #
    #     run_system(config, make, register, refine, integrate, debug_mode)