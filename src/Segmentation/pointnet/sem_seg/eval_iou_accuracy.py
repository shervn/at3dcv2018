import numpy as np
import os

IN_DUMP_DIR = "/usr/stud/tranthi/segmentation/03_repos/pointnet/sem_seg/log/dump"


PRED_FILELIST = IN_DUMP_DIR + "/pred_filelist.txt"
LOG_OUT_NAME = 'log_ioueval.txt'
LOG_FOUT = open(os.path.join(IN_DUMP_DIR, LOG_OUT_NAME), 'w')
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)
log_string('evaluating prediction results of ' + PRED_FILELIST)

pred_data_label_filenames = [line.rstrip() for line in open(PRED_FILELIST)]
gt_label_filenames = [f.rstrip('_pred.txt') + '_gt.txt' for f in pred_data_label_filenames]
num_room = len(gt_label_filenames)


gt_classes = [0 for _ in range(13)]
positive_classes = [0 for _ in range(13)]
true_positive_classes = [0 for _ in range(13)]
for i in range(num_room):
    log_string(i)
    data_label = np.loadtxt(pred_data_label_filenames[i])
    pred_label = data_label[:,-1]
    gt_label = np.loadtxt(gt_label_filenames[i])
    log_string(gt_label.shape)
    for j in xrange(gt_label.shape[0]):
        gt_l = int(gt_label[j])
        pred_l = int(pred_label[j])
        gt_classes[gt_l] += 1
        positive_classes[pred_l] += 1
        true_positive_classes[gt_l] += int(gt_l==pred_l)


log_string(gt_classes)
log_string(positive_classes)
log_string(true_positive_classes)


log_string('Overall accuracy: {0}'.format(sum(true_positive_classes)/float(sum(positive_classes))))

log_string 'IoU:'
iou_list = []
for i in range(13):
    iou = true_positive_classes[i]/float(gt_classes[i]+positive_classes[i]-true_positive_classes[i])
    log_string(iou)
    iou_list.append(iou)

log_string(sum(iou_list)/13.0)
