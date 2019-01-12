import numpy as np
from open3d import *
import copy


class Augmentor:

    def __init__(self, pointcloud):
        print("Augmentor Just Starting...")
        self.pointcloud = pointcloud
        draw_geometries_with_editing([self.pointcloud])