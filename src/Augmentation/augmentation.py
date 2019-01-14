import numpy as np
from open3d import *
import copy
import sys,os

from collections import defaultdict

bed_pc_name = '/Samples/new_bed.ply'
bed_pc_corners = [309941, 156052, 201237, 157628]
bed_index = 15

old_bed_corners = [1229, 2329, 240, 845]

class Augmentor:

    def __init__(self, pointcloud, labeld_pointcloud):
        self.pointcloud = pointcloud
        self.labeld_pointcloud = labeld_pointcloud
        
        self.number_of_objects = 0
        self.color_hash_map = defaultdict(int)

        self.__find_all_objects()

        # for i in range(0, self.number_of_objects):
        #     print(self.number_of_objects - i)
        #     draw_geometries([self.get_object_with_index(self.number_of_objects - i)])

    def get_object_with_index(self, i):
         
        new_pointcloud = select_down_sample(self.pointcloud, self.classfied_objects[i])
        return new_pointcloud

    def remove_object_with_index(self, i):

        l = range(0, len(self.pointcloud.points))
        new_pointcloud = select_down_sample(self.pointcloud, [x for x in l if x not in self.classfied_objects[i]])
        return new_pointcloud
    
    def change_object(self, t):

        scene_without_old_object = self.remove_object_with_index(bed_index)
        old_object = self.get_object_with_index(bed_index)

        transformed_new_object = self.__manual_registration(t, old_object)
        #draw_geometries([transformed_new_object, scene_without_old_object])
        draw_geometries([transformed_new_object, scene_without_old_object])
        return transformed_new_object

    def __manual_registration(self, t, target):

        source = read_point_cloud(bed_pc_name) #ACCORDING TO t
        picked_id_source = bed_pc_corners #ACCORDING TO t

        #picked_id_target = self.__pick_points(target)
        picked_id_target = old_bed_corners
        assert(len(picked_id_source)>=3 and len(picked_id_target)>=3)
        assert(len(picked_id_source) == len(picked_id_target))
        corr = np.zeros((len(picked_id_source),2))
        corr[:,0] = picked_id_source
        corr[:,1] = picked_id_target

        # estimate rough transformation using correspondences
        p2p = TransformationEstimationPointToPoint()
        trans_init = p2p.compute_transformation(source, target,
                Vector2iVector(corr))

        # point-to-point ICP for refinement
        threshold = 0.03 # 3cm distance threshold
        reg_p2p = registration_icp(source, target, threshold, trans_init,
                TransformationEstimationPointToPoint())

        source_temp = copy.deepcopy(source)
        source_temp.transform(reg_p2p.transformation)
        return (source_temp)

    def __run_icp_for_objects(self, source, target, target_without_old_object):
        
        threshold = 0.02
        trans_init = np.asarray(
                    [[0.1, 0.1, -0.1,  0.1],
                    [-0.1, 0.1, -0.1,  0.1],
                    [0.1, 0.1,  0.1, -0.1],
                    [0.1, 0.1, 0.1, 0.1]])
                    
        # trans_init = np.asarray(
        #     [[0.0, 0.0, 0.0, 0.0],
        #     [0.0, 0.0, 0.0, 0.0],
        #     [0.0, 0.0, 0.0, 0.0],
        #     [0.0, 0.0, 0.0, 0.0]])
                    
        evaluation = evaluate_registration(source, target,
                threshold, trans_init)
        reg_p2p = registration_icp(source, target, threshold, trans_init,
                TransformationEstimationPointToPoint())

        source.transform(reg_p2p.transformation)
        
        draw_geometries([source, target_without_old_object])

    def __find_all_objects(self):
        l = defaultdict(list)
        i = 0
        for color in self.labeld_pointcloud.colors:
            l[self.ــget_object_index_by_label_color(color)].append(i)
            i += 1

        self.classfied_objects = []
        for ind in range(-1, self.number_of_objects):
            temp = l[ind]
            self.classfied_objects.append(temp)

        self.classfied_objects.sort(key = lambda x: len(x))

    def ــget_object_index_by_label_color(self, color):
        hashed_color = str(color[0]) + '-' + str(color[1]) + '-' + str(color[2])
        if(color is '0.0-0.0-0.0'):
            return -1
        if(hashed_color in self.color_hash_map):
            return self.color_hash_map[hashed_color]
        else:
            self.color_hash_map[hashed_color] = self.number_of_objects
            self.number_of_objects += 1
            return self.color_hash_map[hashed_color]

    def __pick_points(self, pcd):
        print("")
        print("1) Please pick at least three correspondences using [shift + left click]")
        print("   Press [shift + right click] to undo point picking")
        print("2) Afther picking points, press q for close the window")
        vis = VisualizerWithEditing()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.run() # user picks points
        vis.destroy_window()
        print("")
        return vis.get_picked_points()

if __name__ == "__main__":

    rel_path = os.path.realpath('')
    l = read_point_cloud(rel_path + '/Samples/scene0000_01_vh_clean_2.labels.ply')
    r = read_point_cloud(rel_path + '/Samples/scene0000_01_vh_clean_2.ply')
    bed_pc_name = rel_path + bed_pc_name

    t = Augmentor(r, l)

    t.change_object(bed_index)