import numpy as np
from open3d import *
import copy
from collections import defaultdict
import sys

from paths import augmentation_util
sys.path.append(augmentation_util)

from utils import helper
from utils.icp_helper import get_registeration
from utils.vis_helper import *

from config import *
from paths import *

class Augmentor:

    def __init__(self, pointcloud, labeld_pointcloud):

        self.pointcloud = pointcloud
        self.labeld_pointcloud = labeld_pointcloud
        
        self.__find_all_objects()

    def __show_all_objects(self):

        for t in self.objects_dictionary_by_color:
            pcd = self.get_object_with_hashed_color(t)
            show_pcd([pcd])

    def get_object_with_hashed_color(self, hashed_color):
        return select_down_sample(self.pointcloud, self.objects_dictionary_by_color[hashed_color])

    def remove_object_with_index(self, hashed_color):

        l = list(range(0, len(self.pointcloud.points)))
        print(len(l))
        for x in self.objects_dictionary_by_color[hashed_color]:
                l.remove(x)

        u = select_down_sample(self.pointcloud, l)
        return u
    
    def change_object(self, object_to_change, source_pointcloud):

        object_to_change_hash = self.__get_hashed_color_of_object_from_name(object_to_change)

        scene_without_old_object = self.remove_object_with_index(object_to_change_hash)
        old_object = self.get_object_with_hashed_color(object_to_change_hash)

        transformed_new_object = self.__automated_registraion(source_pointcloud, old_object)

        return transformed_new_object, scene_without_old_object

    def __get_hashed_color_of_object_from_name(self, object_name):

        return objects_hash[object_name] 

    def __get_new_object_from_dataset(self, object_name, index=0):

        return read_point_cloud(new_objects_address[object_name][index])

    def __automated_registraion(self, source, target):

        transformation = get_registeration(0.5, source, target)

        source_temp = copy.deepcopy(source)
        source_temp.transform(transformation)

        return (source_temp)

    def __manual_registration(self, source, target):
        #not using. could be the next best thing to do.

        picked_id_source = bed_pc_corners #ACCORDING TO t

        picked_id_target = self.__pick_points(target)
        #picked_id_target = old_bed_corners
        print(picked_id_target)

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

    def __change_color(self, object, color):
        object_tmp = copy.deepcopy(object)
        object_tmp.paint_uniform_color(color)

        return object_tmp

    def __find_all_objects(self):

        l = defaultdict(list)
        i = 0
        for color in self.labeld_pointcloud.colors:
            l[self.ــget_object_hash_by_label_color(color)].append(i)
            i += 1

        self.objects_dictionary_by_color = l


    def ــget_object_hash_by_label_color(self, color):

        hashed_color = str(color[0]) + '-' + str(color[1]) + '-' + str(color[2])
        return hashed_color

    def __pick_points(self, pcd):
        #only needed when using __manul_registration

        print("")
        print("1) Please pick at least three correspondences using [shift + left click]")
        print("   Press [shift + right click] to undo point picking")
        print("2) Afther picking points, press q for close the window")

        vis = VisualizerWithEditing()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.run()
        vis.destroy_window()

        return vis.get_picked_points()