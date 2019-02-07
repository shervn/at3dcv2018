
from open3d import *
import sys
from paths import reconstruction_system, reconstruction_config, sensors, camera_trajectory_path
sys.path.append(reconstruction_system)
from run_system import run_system



class Reconstructor:

    def __init__(self):

        # # record a scene using realsense
        # record()

        # perform 3d reconstruction
        config = reconstruction_config
        make = True
        register = True
        refine = True
        integrate = True
        debug_mode = False

        self.reconstructed_pointcloud = run_system(config, make, register, refine, integrate, debug_mode)

        # name = "/home/pti/Downloads/tum/at3dcv/project/pointclouds_for_fun/with_chair.ply"
        name1 = "salala.ply"
        # self.reconstructed_pointcloud = read_point_cloud(name)
        draw_geometries([self.reconstructed_pointcloud])

        # write_triangle_mesh("integrated", self.reconstructed_pointcloud, False, True)
        print("3d reconstruction finished")



# outlier removal


        # print("Load a ply point cloud, print it, and render it")
        # pcd = read_point_cloud("/home/pti/Downloads/tum/at3dcv/project/at3dcv2018/src/Reconstruction/ReconstructionSystem/dataset/realsense/scene/integrated.ply")
        # # draw_geometries([pcd])
        # print("Downsample the point cloud with a voxel of 0.02")
        # voxel_down_pcd = voxel_down_sample(pcd, voxel_size=0.008)
        # # draw_geometries([voxel_down_pcd])
        #
        # # print("Every 5th points are selected")
        # # uni_down_pcd = uniform_down_sample(pcd, every_k_points=5)
        # # draw_geometries([uni_down_pcd])
        #
        # print("Statistical oulier removal")
        # cl, ind = statistical_outlier_removal(voxel_down_pcd,
        #                                       nb_neighbors=200, std_ratio=1.5)
        #
        # # print("Radius oulier removal")
        # # cl, ind = radius_outlier_removal(voxel_down_pcd,
        # #                                  nb_points=16, radius=0.05)
        #
        # inlier_cloud = select_down_sample(voxel_down_pcd, ind)
        # outlier_cloud = select_down_sample(voxel_down_pcd, ind, invert=True)
        #
        # print("Showing outliers (red) and inliers (gray): ")
        # outlier_cloud.paint_uniform_color([1, 0, 0])
        # # inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
        # # draw_geometries([inlier_cloud, outlier_cloud])
        # # draw_geometries([inlier_cloud])
        #
        # # write_triangle_mesh("sampled", inlier_cloud, False, True)
        # write_point_cloud(name1, inlier_cloud, False, True)

