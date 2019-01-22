import numpy as np
from open3d import *

import argparse


def create_datafile_txt(pc, name, downsample = 1):

    txt = ''
    number_of_points = len(pc.points)

    txt += str(number_of_points)
    for i in range(0, number_of_points, downsample):
        txt += '\n'
        print(i)
        for j in range(0, 3):
            txt += str(round(pc.points[i][j], 5))
            txt += ' '

    text_file = open(name, "w")
    text_file.write(txt)
    text_file.close()

def make_pc_ready_for_pycpd(pc, downsample):

    number_of_points = (int)(len(pc.points)/downsample)
    points = np.zeros((number_of_points, 3))

    for i in range(1, number_of_points):
        points[i - 1, :] = pc.points[i * downsample - 1]
    
    return points

def downsample_pc(pc, voxel_size = 0.05):
    downpcd = voxel_down_sample(pc, voxel_size)
    return downpcd