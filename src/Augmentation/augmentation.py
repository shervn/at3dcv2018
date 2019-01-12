import numpy as np
from open3d import *
import copy


class Augmentor:

    def __init__(self, pointcloud):
        print("Augmentor Just Starting...")
        self.pointcloud = pointcloud
        draw_geometries_with_editing([self.pointcloud])


    def display_inlier_outlier(cloud, ind):
        inlier_cloud = select_down_sample(cloud, ind)
        outlier_cloud = select_down_sample(cloud, ind, invert=True)

        print("Showing outliers (red) and inliers (gray): ")
        outlier_cloud.paint_uniform_color([1, 0, 0])
        inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
        draw_geometries([inlier_cloud, outlier_cloud])

    def demo_crop_geometry(pcd):
        print("Demo for manual geometry cropping")
        print("1) Press 'Y' twice to align geometry with negative direction of y-axis")
        print("2) Press 'K' to lock screen and to switch to selection mode")
        print("3) Drag for rectangle selection,")
        print("   or use ctrl + left click for polygon selection")
        print("4) Press 'C' to get a selected geometry and to save it")
        print("5) Press 'F' to switch to freeview mode")
        
        cutpcd = draw_geometries_with_editing([pcd])
        return cutpcd

    def demo_crop_geometry():
        print("Demo for manual geometry cropping")
        print("1) Press 'Y' twice to align geometry with negative direction of y-axis")
        print("2) Press 'K' to lock screen and to switch to selection mode")
        print("3) Drag for rectangle selection,")
        print("   or use ctrl + left click for polygon selection")
        print("4) Press 'C' to get a selected geometry and to save it")
        print("5) Press 'F' to switch to freeview mode")
        pcd = read_point_cloud("../../TestData/ICP/cloud_bin_0.pcd")
        draw_geometries_with_editing([pcd])

    def draw_registration_result(source, target, transformation):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(transformation)
        draw_geometries([source_temp, target_temp])

    def pick_points(pcd):
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

    def demo_manual_registration(bedpclname, roompclname):
        print("Demo for manual ICP")
        print("Visualization of two point clouds before manual alignment")

        source = read_point_cloud(roompclname)
        target = read_point_cloud(bedpclname)

        draw_registration_result(source, target, np.identity(4))

        # pick points from two point clouds and builds correspondences
        picked_id_source = [357967, 121944, 203580, 157324]
        picked_id_target = pick_points(target)
        assert(len(picked_id_source)>=3 and len(picked_id_target)>=3)
        assert(len(picked_id_source) == len(picked_id_target))
        corr = np.zeros((len(picked_id_source),2))
        corr[:,0] = picked_id_source
        corr[:,1] = picked_id_target

        # estimate rough transformation using correspondences
        print("Compute a rough transform using the correspondences given by user")
        p2p = TransformationEstimationPointToPoint()
        trans_init = p2p.compute_transformation(source, target,
                Vector2iVector(corr))

        # point-to-point ICP for refinement
        print("Perform point-to-point ICP refinement")
        threshold = 0.03 # 3cm distance threshold
        reg_p2p = registration_icp(source, target, threshold, trans_init,
                TransformationEstimationPointToPoint())
        draw_registration_result(source, target, reg_p2p.transformation)
        print("")
