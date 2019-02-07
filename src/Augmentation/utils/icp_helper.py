import numpy as np
from open3d import *

import argparse

def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = voxel_down_sample(pcd, voxel_size)

    radius_normal = voxel_size * 2
    estimate_normals(pcd_down, KDTreeSearchParamHybrid(
            radius = radius_normal, max_nn = 30))

    radius_feature = voxel_size * 5
    pcd_fpfh = compute_fpfh_feature(pcd_down,
            KDTreeSearchParamHybrid(radius = radius_feature, max_nn = 100))
            
    return pcd_down, pcd_fpfh

def prepare_dataset(source, target, voxel_size):

    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0],
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0]])

    source.transform(trans_init)

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

    return source_down, target_down, source_fpfh, target_fpfh

def execute_global_registration(
        source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.0

    result = registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh,
            distance_threshold,
            TransformationEstimationPointToPoint(False), 4,
            [CorrespondenceCheckerBasedOnEdgeLength(0.9),
            CorrespondenceCheckerBasedOnDistance(distance_threshold)],
            RANSACConvergenceCriteria(8000000, 500))
    return result

def refine_registration(source, target, source_fpfh, target_fpfh, result_ransac, voxel_size):

    distance_threshold = voxel_size * 0.4
    result = registration_icp(source, target, distance_threshold,
            result_ransac.transformation,
            TransformationEstimationPointToPoint())
    return result


def get_registeration(voxel_size, source, target):

    source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(source, target, voxel_size)

    result_ransac = execute_global_registration(source_down, target_down,
            source_fpfh, target_fpfh, voxel_size)

    result_icp = refine_registration(source, target,
            source_fpfh, target_fpfh, result_ransac, voxel_size)

    return result_icp.transformation
