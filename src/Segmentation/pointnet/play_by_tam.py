import sys, os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))


from utils_by_tam import *


# INSPECT THE H5 FILE
pointnet_root = '/usr/stud/tranthi/segmentation/03_repos/pointnet'
# pointnet_root = '/Users/tamtran/Desktop/TUM/1819Winter/Praktikum_ATin3DCV/project/03_repos/pointnet'
#
# print '23'
# explore_h5(pointnet_root+'/sem_seg/indoor3d_sem_seg_hdf5_data/ply_data_all_23.h5')
#

# DOES NOT WORK
# print '\n'
# data_json = explore_json(pointnet_root+'/data/modelnet40_ply_hdf5_2048/ply_data_train_3_id2file.json')
# print type(data_json), len(data_json)
# print data_json[:10]


# INSPECT THE JSON FILE
# import json
# from pprint import pprint
#
# with open(pointnet_root+'/data/modelnet40_ply_hdf5_2048/ply_data_train_0_id2file.json') as f:
#     data = json.load(f)
#
# pprint(data)


# INSPECT THE CKPT FILE
# from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
#
# LOG_DIR = pointnet_root + "/sem_seg/log"
# MODEL_PATH = os.path.join(LOG_DIR, "model_files1224.ckpt")
# print_tensors_in_checkpoint_file(MODEL_PATH, all_tensors=True, tensor_name='')



# INSPECT THE LOG FILE
# values = parse_log_file(pointnet_root+'/sem_seg/log/log_train_filesOdd.txt',
#                         ['mean loss', 'accuracy', 'eval mean loss', 'eval accuracy', 'eval avg class acc'])


# get counts of file dimensions in a directory
# sum_dims(pointnet_root + '/data/stanford_indoor3d', 'npy')


# save ply to npy
directory = pointnet_root + '/data/custom_data'
#for file in os.listdir(directory):
for file in ['integrated.ply']:
    if file.endswith('.ply'):
        ply_to_npy(directory + '/' + file, directory + '/' + file.split('.')[0] + '.npy')



# view npy's color distribution
# print('hey')
# #directory = '/usr/stud/tranthi/segmentation/03_repos/pointnet/sem_seg/log/dump_filesEven_tmp'
# directory = '/Users/tamtran/Desktop/TUM/1819Winter/Praktikum_ATin3DCV/project/03_repos/pointnet/sem_seg/log/dump_filesEven_tmp'
# file = '/Area_6_conferenceRoom_1_pred_redOrangeGreenYellow.ply'
# # directory = '/usr/stud/tranthi/Downloads'
# # file = '/chair_downsampled_3.npy'
#
# if file.split('.')[-1] == 'ply':
#     pcd = read_point_cloud(directory+file)
#     n = np.asarray(pcd.points).shape[0]
#     data = np.zeros((n,6))
#     data[:,:3] = pcd.points
#     data[:,3:] = pcd.colors
# if file.split('.')[-1] == 'npy':
#     data = np.load(directory + file)
#     data = np.asarray(data)
# if file.split('.')[-1] == 'txt':
#     f = open(directory + file, 'r')
#     data = f.readlines()
#     data = [[float(x) for x in line.split(' ')] for line in data]
#     data = np.asarray(data)
#
# print(file)
# print(np.histogram(data[:,3], bins=20))
# print(np.histogram(data[:,4], bins=20))
# print(np.histogram(data[:,5], bins=20))
# if data.shape[1]>6:
#     print(np.histogram(data[:,6], bins=20))
#     print(Counter(data[:,7]))

# fig, axes = plt.subplots(3,1)
# axes[0].hist(data[:,3])
# axes[1].hist(data[:,4])
# axes[2].hist(data[:,5])
# plt.show()

