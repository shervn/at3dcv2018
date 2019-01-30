import h5py
import numpy as np
from plyfile import PlyData, PlyElement

#f = h5py.File("./hdf5_data/data_training.h5", 'w')
f = h5py.File("./hdf5_data/data_testing.h5", 'w')

a_data = np.zeros((len(filenames), 2048, 3))
a_pid = np.zeros((len(filenames), 2048), dtype = np.uint8)	

plydata = PlyData.read('/Users/shervindehghani/Desktop/hello/at3dcv2018/Samples/with_chair.ply')
piddata = [line.rstrip() for line in open("./points_label/" + filenames[i] + ".seg", 'r')]
for j in range(0, 2048):
	a_data[i, j] = [plydata['vertex']['x'][j], plydata['vertex']['y'][j], plydata['vertex']['z'][j]]
	a_pid[i,j] = piddata[j]

data = f.create_dataset("data", data = a_data)
pid = f.create_dataset("pid", data = a_pid)