import numpy as np
from open3d import *
import h5py
from collections import Counter

POINTNET_ROOT = "/usr/stud/tranthi/segmentation/03_repos/pointnet"
FILE_NAME = "Area_6_copyRoom_1"


if True:
    # handle original/rgb
    NPY_IN_FILE = POINTNET_ROOT + "/data/stanford_indoor3d/" + FILE_NAME + ".npy"
    data = np.load(NPY_IN_FILE) # 7 channels, last one maybe class index
    data = np.asarray(data)
    data[:, 3:6] = data[:, 3:6] / 255  # 0 to 1 range
    print(NPY_IN_FILE.split('/')[-1], data.shape) #(1136617, 7)

    # handle ground truth
    label = data[:,6]
    print('gt label', Counter(label))
    labelc = np.zeros((data.shape[0], 3))
    labelc[:,0] = label / np.max(label)
    labelc[:,1] = (label / np.max(label))**1
    labelc[:,2] = (label / np.max(label))**1

    # handle prediction
    PRED_IN_FILE = POINTNET_ROOT + "/sem_seg/log/dump/" + FILE_NAME + "_pred.txt"
    pred_f = open(PRED_IN_FILE, 'r')
    pred_data = pred_f.readlines()
    pred_data = [[float(x) for x in line.split(" ")] for line in pred_data]
    pred_data = np.asarray(pred_data)
    print(PRED_IN_FILE.split('/')[-1], pred_data.shape)

    pred_label = pred_data[:,-1]
    print('pred label', Counter(pred_label))
    pred_labelc = np.zeros((pred_data.shape[0], 3))
    pred_labelc[:,0] = pred_label / np.max(pred_label)
    pred_labelc[:,1] = (pred_label / np.max(pred_label))**1
    pred_labelc[:,2] = (pred_label / np.max(pred_label))**1

pcd_og = PointCloud()
pcd_og.points = Vector3dVector(data[:,:3])
if data.shape[-1]>3: pcd_og.colors = Vector3dVector(data[:,3:6])

pcd_gt = PointCloud()
pcd_gt.points = Vector3dVector(data[:,:3])
if data.shape[-1]>6: pcd_gt.colors = Vector3dVector(labelc)

pcd_pred = PointCloud()
pcd_pred.points = Vector3dVector(pred_data[:,:3])
pcd_pred.colors = Vector3dVector(pred_labelc)

draw_geometries([pcd_gt])
draw_geometries([pcd_pred])