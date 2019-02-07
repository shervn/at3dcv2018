# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Tutorial/ReconstructionSystem/register_fragments.py

import numpy as np
from open3d import *
import sys

from paths import utility
sys.path.append(utility)
from file import *

from visualization import *
from optimize_posegraph import *
from refine_registration import *


def preprocess_point_cloud(pcd, config):
    voxel_size = config["voxel_size"]
    pcd_down = voxel_down_sample(pcd, voxel_size)
    estimate_normals(pcd_down,
            KDTreeSearchParamHybrid(radius = voxel_size * 2.0, max_nn = 30))
    pcd_fpfh = compute_fpfh_feature(pcd_down,
            KDTreeSearchParamHybrid(radius = voxel_size * 5.0, max_nn = 100))
    return (pcd_down, pcd_fpfh)


def register_point_cloud_fpfh(source, target,
        source_fpfh, target_fpfh, config):
    distance_threshold = config["voxel_size"] * 1.4
    if config["global_registration"] == "fgr":
        result = registration_fast_based_on_feature_matching(
                source, target, source_fpfh, target_fpfh,
                FastGlobalRegistrationOption(
                maximum_correspondence_distance = distance_threshold))
    if config["global_registration"] == "ransac":
        result = registration_ransac_based_on_feature_matching(
                source, target, source_fpfh, target_fpfh,
                distance_threshold,
                TransformationEstimationPointToPoint(False), 4,
                [CorrespondenceCheckerBasedOnEdgeLength(0.9),
                CorrespondenceCheckerBasedOnDistance(distance_threshold)],
                RANSACConvergenceCriteria(4000000, 500))
    if (result.transformation.trace() == 4.0):
        return (False, np.identity(4), np.zeros((6,6)))
    information = get_information_matrix_from_point_clouds(
            source, target, distance_threshold, result.transformation)
    if information[5,5] / min(len(source.points),len(target.points)) < 0.3:
        return (False, np.identity(4), np.zeros((6,6)))
    return (True, result.transformation, information)


def compute_initial_registration(s, t, source_down, target_down,
        source_fpfh, target_fpfh, path_dataset, config):

    if t == s + 1: # odometry case
        print("Using RGBD odometry")
        pose_graph_frag = read_pose_graph(join(path_dataset,
                config["template_fragment_posegraph_optimized"] % s))
        n_nodes = len(pose_graph_frag.nodes)
        transformation_init = np.linalg.inv(
                pose_graph_frag.nodes[n_nodes-1].pose)
        (transformation, information) = \
                multiscale_icp(source_down, target_down,
                [config["voxel_size"]], [50], config, transformation_init)
    else: # loop closure case
        (success, transformation, information) = register_point_cloud_fpfh(
                source_down, target_down,
                source_fpfh, target_fpfh, config)
        if not success:
            print("No resonable solution. Skip this pair")
            return (False, np.identity(4), np.zeros((6,6)))
    print("transformation")
    print(transformation)

    if config["debug_mode"]:
        print("register")
        draw_registration_result(source_down, target_down,
                transformation)

    return (True, transformation, information)


def update_posegrph_for_scene(s, t, transformation, information,
        odometry, pose_graph):
    if t == s + 1: # odometry case
        odometry = np.dot(transformation, odometry)
        odometry_inv = np.linalg.inv(odometry)
        pose_graph.nodes.append(PoseGraphNode(odometry_inv))
        pose_graph.edges.append(
                PoseGraphEdge(s, t, transformation,
                information, uncertain = False))
    else: # loop closure case
        pose_graph.edges.append(
                PoseGraphEdge(s, t, transformation,
                information, uncertain = True))
    return (odometry, pose_graph)


def register_point_cloud_pair(ply_file_names, s, t, config):
    print("reading %s ..." % ply_file_names[s])
    source = read_point_cloud(ply_file_names[s])
    print("reading %s ..." % ply_file_names[t])
    target = read_point_cloud(ply_file_names[t])
    (source_down, source_fpfh) = preprocess_point_cloud(source, config)
    (target_down, target_fpfh) = preprocess_point_cloud(target, config)
    (success, transformation, information) = \
            compute_initial_registration(
            s, t, source_down, target_down,
            source_fpfh, target_fpfh, config["path_dataset"], config)
    if t != s + 1 and not success:
        return (False, np.identity(4), np.identity(6))
    if config["debug_mode"]:
        print(transformation)
        print(information)
    return (True, transformation, information)


# other types instead of class?
class matching_result:
    def __init__(self, s, t):
        self.s = s
        self.t = t
        self.success = False
        self.transformation = np.identity(4)
        self.infomation = np.identity(6)


def make_posegraph_for_scene(ply_file_names, config):
    pose_graph = PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(PoseGraphNode(odometry))

    n_files = len(ply_file_names)
    matching_results = {}
    for s in range(n_files):
        for t in range(s + 1, n_files):
            matching_results[s * n_files + t] = matching_result(s, t)

    if config["python_multi_threading"]:
        from joblib import Parallel, delayed
        import multiprocessing
        import subprocess
        MAX_THREAD = min(multiprocessing.cpu_count(),
                max(len(matching_results), 1))
        results = Parallel(n_jobs=MAX_THREAD)(
                delayed(register_point_cloud_pair)(ply_file_names,
                matching_results[r].s, matching_results[r].t, config)
                for r in matching_results)
        for i, r in enumerate(matching_results):
            matching_results[r].success = results[i][0]
            matching_results[r].transformation = results[i][1]
            matching_results[r].information = results[i][2]
    else:
        for r in matching_results:
            (matching_results[r].success, matching_results[r].transformation,
                    matching_results[r].information) = \
                    register_point_cloud_pair(ply_file_names,
                    matching_results[r].s, matching_results[r].t, config)

    for r in matching_results:
        if matching_results[r].success:
            (odometry, pose_graph) = update_posegrph_for_scene(
                    matching_results[r].s, matching_results[r].t,
                    matching_results[r].transformation,
                    matching_results[r].information,
                    odometry, pose_graph)
    write_pose_graph(join(config["path_dataset"],
            config["template_global_posegraph"]), pose_graph)


def run(config):
    print("register fragments.")
    set_verbosity_level(VerbosityLevel.Debug)
    ply_file_names = get_file_list(join(
            config["path_dataset"], config["folder_fragment"]), ".ply")
    make_clean_folder(join(config["path_dataset"], config["folder_scene"]))
    make_posegraph_for_scene(ply_file_names, config)
    optimize_posegraph_for_scene(config["path_dataset"], config)
