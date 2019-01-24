from open3d import *

labled_name = '../Samples/scene0000_00_vh_clean_2.labels.ply'

class Segmenter:
    def __init__(self, raw_pointcloud):
        self.raw_pointcloud = raw_pointcloud
        self.__segment()

    def __segment(self):
        #do the magic
        self.labled_pointcloud = read_point_cloud(labled_name)
    
