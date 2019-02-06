import sys, os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))


from utils_by_tam import *
pointnet_root = '/usr/stud/tranthi/segmentation/03_repos/pointnet2'

# print 'hey'
# explore_h5(pointnet_root+'/data/modelnet40_ply_hdf5_2048/ply_data_train2.h5')

#print '\n'
# data_json = explore_json(pointnet_root+'/data/modelnet40_ply_hdf5_2048/ply_data_train_3_id2file.json')
# print type(data_json), len(data_json)
# print data_json[:10]


# VIEW JSON FILE... MAYBE WORKS
# import json
from pprint import pprint
#
# with open(pointnet_root+'/data/modelnet40_ply_hdf5_2048/ply_data_train_2_id2file.json') as f:
#     data = json.load(f)
#
# print(len(data))
# pprint(data[:100])



values = parse_log_file(pointnet_root + '/log/log_train_scan_trial0.txt')
