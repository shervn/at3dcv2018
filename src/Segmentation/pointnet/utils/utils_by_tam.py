import sys, os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

import numpy as np
import json
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from open3d import *
from open3d.open3d.geometry import read_point_cloud
from open3d.open3d.geometry import write_point_cloud
from collections import Counter

def explore_h5(hdf_file):

    """Traverse all datasets across all groups in HDF5 file."""

    import h5py

    def h5py_dataset_iterator(g, prefix=''):
        for key in g.keys():
            item = g[key]
            path = '{}/{}'.format(prefix, key)
            if isinstance(item, h5py.Dataset): # test for dataset
                yield (path, item)
            elif isinstance(item, h5py.Group): # test for group (go down)
                for h in h5py_dataset_iterator(item, path):
                    yield h

    with h5py.File(hdf_file, 'r') as f:
        for (path, dset) in h5py_dataset_iterator(f):
            print(path, dset)

            groups = ['data', 'faceId', 'label', 'try']
            for g in groups:
                if g in path:
                    print(np.array(f[g]))

    return None



# bad, need to fix. it clears all data in .json file
def explore_json(json_file):
    data=[]
    with open(json_file, "w") as f:
        json.dump(data, f)
    return data


def parse_log_file(log_file, items):
    # parse and plot
    # values = list of items
    file = open(log_file, 'r')
    values = {}
    for i in items:
        values[i] = []

    for l in file:
        precolon = l.split(':')[0]
        postcolon = l.split(':')[-1]
        if precolon in items:
            values[precolon].append(float(postcolon))

    plt.figure()
    for i,v in values.items():
        plt.plot(v, label=i)
        for a,b in zip(range(len(v)), v):
            if a%5==0:
                plt.text(a,b,str(b)[:4])
    plt.title(log_file.split('/')[-1])
    plt.legend()

    plt.show()

    return values



def sum_dims(directory, file_type):

    shape_total = np.zeros((1,2))
    for file in os.listdir(directory):
        if file.endswith('.'+file_type):
            #shape = np.asarray(np.asarray(np.load(directory + '/' + file)).shape())
            shape = np.asarray(np.asarray(np.load(directory + '/' + file)).shape)
            print(shape)
            shape_total += shape


            data = np.load(directory + '/' + file)
            data = np.asarray(data)
            shape = data.shape
            shape = np.asarray(shape)
            print(shape)
            shape_total += shape

    print('shape_total', shape_total)
    return shape_total


def ply_to_npy(in_ply_file, out_npy_file):
    # gives fake label
    ply = read_point_cloud(in_ply_file)
    n = np.asarray(ply.points).shape[0]
    npy = np.zeros((n, 7))
    npy[:,:3] = np.asarray(ply.points)
    npy[:,3:6] = np.asarray(ply.colors)

    np.save(out_npy_file, npy)



def save_to_ply(in_file, out_file, label_color=True):

    if in_file.split('.')[-1] == 'txt':
        f = open(in_file, 'r')
        data = f.readlines()
        data = [[float(x) for x in line.split(" ")] for line in data]
        data = np.asarray(data)

        pcd = PointCloud()
        pcd.points = Vector3dVector(data[:,:3])
        if label_color:
            label = data[:,7]
            labelc = label_to_RGB(label)
            pcd.colors = Vector3dVector(labelc)
        else:
            pcd.colors = Vector3dVector(data[:,3:6])
        write_point_cloud(out_file, pcd)
        print('saved', out_file)


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
