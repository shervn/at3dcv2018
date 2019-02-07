'''
visualize point clouds of many file types. can set whether to view rgb or label of data
'''




import numpy as np
from open3d import *
# from open3d.geometry import write_point_cloud
import h5py
from collections import Counter
# from utils_by_tam import *

file_type = 'txt' # h5, txt, npy, ply
show_color = 0
show_label = 1
show_custom = 0
assert(show_color + show_label + show_custom <= 1)

POINTNET_ROOT = ''
#POINTNET_ROOT = "/usr/stud/tranthi/segmentation/03_repos/pointnet"
# IN_FILE = ''

def label_to_RGB(label):

    mapping = {0.0: [255,0,0], #ceiling #red
               1.0: [255,128,0], #floor #orange -> yellow
               2.0: [255,255,0], #wall #yellow
               3.0: [0,255,0], #beam #green
               4.0: [0,255,255], #column #skyblue
               5.0: [0,0,255], #window #blue
               6.0: [128,0,255], #door #purple -> magenta
               7.0: [255,0,255], #table #pink -> magenta
               8.0: [102,0,0], #chair #brown -> red
               9.0: [102,102,0], #sofa #green-brown -> yellow
               10.0: [204,255,204], #bookcase #pastel-green -> white
               11.0: [255,204,255], #board #pinkish-lavender -> white
               12.0: [64,64,64] #clutter #dark-grey -> white
               }

    labelc = [mapping[l] for l in label]
    labelc = np.asarray(labelc)
    return labelc

if file_type == 'h5':
    IN_FILE = POINTNET_ROOT + "/sem_seg/indoor3d_sem_seg_hdf5_data/ply_data_all_0.h5" # rectangular objs don't quite make sense
    #IN_FILE = POINTNET_ROOT + "/data/modelnet40_ply_hdf5_2048/ply_data_train4.h5" # a bunch of spheres w/o color...

    f = h5py.File(IN_FILE)
    print('h5 file keys', [k for k in f.keys()])
    data = f['data'][:] # 3dim data with 9 channels
    label = f['label'][:] # 2dim with class indexes
    f.close()

    # flatten data to be two-dim
    data = data.reshape(-1,data.shape[-1])
    label = label.reshape(-1,1).squeeze()


if file_type == 'txt':
    #IN_FILE = POINTNET_ROOT + "/sem_seg/log/dump_filesEven_tmp/Area_6_conferenceRoom_1_pred.txt"
    IN_FILE = POINTNET_ROOT + "sem_seg/log/dump_filesEven_voxel2/integrated_pred.txt"
    #IN_FILE = POINTNET_ROOT + "/sem_seg/log/dump/Area_6_copyRoom_1_pred.txt"

    f = open(IN_FILE, 'r') # 6 channels
    data = f.readlines()
    data = [[float(x) for x in line.split(" ")] for line in data]
    data = np.asarray(data)
    data[:,3:6] = data[:,3:6]/255 # 0 to 1 range
    print(data.shape) #(1136617, 6)


if file_type != 'ply':
    pcd = PointCloud()
    pcd.points = Vector3dVector(data[:,:3])
if show_color:
    pcd.colors = Vector3dVector(data[:,3:6])
if show_label:
    assert (data.shape[-1] > 6)
    if 'label' not in dir(): label = data[:, -1]
    print(Counter(label))
    labelc = Vector3dVector(label_to_RGB(label))
    pcd.colors = labelc
if show_custom:
    assert (data.shape[-1] > 6)
    if 'label' not in dir(): label = data[:, -1]
    n = label.shape[0]
    labelc = np.ones((n,3))
    labelc[:int(n/4),:] = np.asarray([255,0,0])
    labelc[int(n/4):int(n/2),:] = np.asarray([255,128,0])
    labelc[int(n/2):int(n/1.33),:] = np.asarray([0,255,0])
    labelc[int(n/1.33):,:] = np.asarray([255,255,0])
    pcd.colors = Vector3dVector(labelc/255)

# OUT_FILE = IN_FILE.split('.')[0] + '_labelc.ply'
# write_point_cloud(OUT_FILE, pcd)
# print('saved', OUT_FILE)
draw_geometries([pcd])
