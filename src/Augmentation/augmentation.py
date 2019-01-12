import numpy as np
from open3d import *
import copy

from collections import defaultdict

bedpclname = '../Samples/cropped.ply'
bed_index = 13

class Augmentor:

    def __init__(self, pointcloud, labeld_pointcloud):
        print("Augmentor Just Starting...") 
        print(bed_index)
        self.pointcloud = pointcloud
        self.labeld_pointcloud = labeld_pointcloud
        
        self.number_of_objects = 0
        self.color_hash_map = defaultdict(int)

        self.__find_all_objects()

        #self.make_object_with_index(self.number_of_objects - 9)

        self.remove_object_with_index(bed_index)

    def __find_all_objects(self):
        l = defaultdict(list)
        i = 0
        for color in self.labeld_pointcloud.colors:
            l[self.get_object_index_by_label_color(color)].append(i)
            i += 1

        self.classfied_objects = []
        for ind in range(-1, self.number_of_objects):
            temp = l[ind]
            self.classfied_objects.append(temp)

        self.classfied_objects.sort(key = lambda x: len(x))

    def get_object_with_index(self, i):
        new_pointcloud = select_down_sample(self.pointcloud, self.classfied_objects[i])
        return new_pointcloud

    def remove_object_with_index(self, i):
        l = range(0, len(self.pointcloud.points))
        new_pointcloud = select_down_sample(self.pointcloud, [x for x in l if x not in self.classfied_objects[i]])
        return new_pointcloud

    def get_object_index_by_label_color(self, color):
        hashed_color = str(color[0]) + '-' + str(color[1]) + '-' + str(color[2])
        if(color is '0.0-0.0-0.0'):
            return -1
        if(hashed_color in self.color_hash_map):
            return self.color_hash_map[hashed_color]
        else:
            self.color_hash_map[hashed_color] = self.number_of_objects
            self.number_of_objects += 1
            return self.color_hash_map[hashed_color]
    
    def change_object(self, t):

        new_object = read_point_cloud(bedpclname) #ACCORDING TO t
        scene_without_old_object = self.remove_object_with_index(bed_index)
        old_object = self.get_object_with_index(bed_index)

        # trans_init = np.asarray(
        #     [[0.0, 0.0, 0.0, 0.0],
        #     [0.0, 0.0, 0.0, 0.0],
        #     [0.0, 0.0, 0.0, 0.0],
        #     [0.0, 0.0, 0.0, 0.0]])

        # draw_registration_result(old_object, new_object, trans_init, scene_without_old_object)
        self.run_icp_for_objects(new_object, old_object, scene_without_old_object)

    def run_icp_for_objects(self, source, target, target_without_old_object):
        
        threshold = 0.02
        trans_init = np.asarray(
                    [[0.862, 0.011, -0.507,  0.5],
                    [-0.139, 0.967, -0.215,  0.7],
                    [0.487, 0.255,  0.835, -1.4],
                    [0.0, 0.0, 0.0, 1.0]])
                    
        evaluation = evaluate_registration(source, target,
                threshold, trans_init)
        reg_p2p = registration_icp(source, target, threshold, trans_init,
                TransformationEstimationPointToPoint())

        source.transform(reg_p2p.transformation)
        draw_geometries([source, target_without_old_object])

if __name__ == "__main__":

    ll = read_point_cloud('../Samples/cropped.ply')
    l = read_point_cloud('../Samples/scene0000_00_vh_clean_2.labels.ply')
    r = read_point_cloud('../Samples/scene0000_00_vh_clean_2.ply')

    t = Augmentor(r, l)
    t.change_object(13)