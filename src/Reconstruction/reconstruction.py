from open3d import *

pointcloud_name = '../Samples/scene0000_00_vh_clean_2.ply'

class Reconstructor:

    def __init__(self):
        #only for testing
        self.reconstructed_pointcloud = read_point_cloud(pointcloud_name)