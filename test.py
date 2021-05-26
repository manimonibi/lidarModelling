import open3d as o3d
from open3d import *
import numpy as np
import pandas
import plotly.graph_objects as go
import os
from open3d.cpu.pybind import geometry
import cv2
import matplotlib
import copy
import time

# load all data to list
clouds = []
directory = os.path.dirname('pairwise/')
for file in sorted(os.listdir(directory)):
    print(file)
    directoryPath = directory
    filePath = os.path.join(directoryPath, file)
    print(filePath)
    clouds.append(o3d.io.read_point_cloud(filePath))

# set params
VOXEL_SIZE = 0.05
CLOUD_INDEX = 1
cloud = clouds[CLOUD_INDEX]
cloud2 = clouds[CLOUD_INDEX + 1]

# # downsample to increase efficiency and speed
# cloud = cloud.voxel_down_sample(VOXEL_SIZE)
# cloud2 = cloud2.voxel_down_sample(VOXEL_SIZE)
#
# # estimate normals
# cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
# cloud2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
# # set to np array
# points = np.asarray(cloud.points)
# points2 = np.asarray(cloud2.points)
#
# # define point colours
# colors = None
# if cloud.has_colors():
#     colors = np.asarray(cloud.colors)
# elif cloud.has_normals():
#     colors = (0.5, 0.5, 0.5) + np.asarray(cloud.normals) * 0.5
# else:
#     geometry.paint_uniform_color((1.0, 0.0, 0.0))
#     colors = np.asarray(geometry.colors)


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])


def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def prepare_dataset(pcd1, pcd2, voxel_size):
    print(":: Load two point clouds and disturb initial pose.")
    source = pcd1
    target = pcd2
    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)
    draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh

source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(cloud, cloud2, VOXEL_SIZE)

def execute_fast_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    print(":: Apply fast global registration with distance threshold %.3f" \
            % distance_threshold)
    result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result

start = time.time()
result_fast = execute_fast_global_registration(source_down, target_down,
                                               source_fpfh, target_fpfh,
                                               VOXEL_SIZE)
print("Fast global registration took %.3f sec.\n" % (time.time() - start))
print(result_fast)
draw_registration_result(source_down, target_down, result_fast.transformation)