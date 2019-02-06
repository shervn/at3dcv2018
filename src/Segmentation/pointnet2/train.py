'''
    Single-GPU training.
    Will use H5 dataset in default. If using normal, will shift to the normal dataset.
'''
import argparse
import math
from datetime import datetime
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider
import tf_util
import modelnet_dataset
import modelnet_h5_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet2_sem_seg', help='Model name [default: pointnet2_sem_seg]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=251, help='Epoch to run [default: 251]')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 16]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--normal', action='store_true', help='Whether to use normal information')
FLAGS = parser.parse_args()

EPOCH_CNT = 0

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_OUT_NAME = 'log_train_trial1.txt' # TODO ADJUST
LOG_FOUT = open(os.path.join(LOG_DIR, LOG_OUT_NAME), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()



NUM_CLASSES = 40
# Shapenet official train/test split (for classification it seems)
if FLAGS.normal:
    assert(NUM_POINT<=10000)
    DATA_PATH = os.path.join(ROOT_DIR, 'data/modelnet40_normal_resampled')
    TRAIN_DATASET = modelnet_dataset.ModelNetDataset(root=DATA_PATH, npoints=NUM_POINT, split='train', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)
    TEST_DATASET = modelnet_dataset.ModelNetDataset(root=DATA_PATH, npoints=NUM_POINT, split='test', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)
else:
    assert(NUM_POINT<=2048)
    TRAIN_DATASET = modelnet_h5_dataset.ModelNetH5Dataset(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'), batch_size=BATCH_SIZE, npoints=NUM_POINT, shuffle=True)
    TEST_DATASET = modelnet_h5_dataset.ModelNetH5Dataset(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'), batch_size=BATCH_SIZE, npoints=NUM_POINT, shuffle=False)


###################################################################
# TODO to adjust (data import and splitting)
# get dataset for indoor scenes, for which you can a class for the dataset type
# so that batch_data, batch_label = TRAIN_DATASET.next_batch(augment=True) and TRAIN_DATASET.reset() are possible






# NUM_POINT = 4096 # resolution of each PC?
# NUM_CLASSES = 13
from collections import Counter
MODEL_OUT_NAME = "model_trial1.ckpt"
if LOG_OUT_NAME.split('_')[-1].split('.')[0] != MODEL_OUT_NAME.split('_')[-1].split('.')[0]:
    raise ValueError('naming of model file and log file seem to be different')
RUN_DESCRIPTION = "model.ckpt and log_train.txt was for ModelNet40 normal for 5 batches \n" \
                  "trial1 is for incorporating the indoor dataset from PointNet1 and for 5 batches \n"
###################################################################


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate        

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl, smpws_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT, 3) #edit num_channels: sem segm reqs 3 channels
            is_training_pl = tf.placeholder(tf.bool, shape=())
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter
            # for you every time it trains.
            batch = tf.get_variable('batch', [],
                initializer=tf.constant_initializer(0), trainable=False)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss 
            pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, NUM_CLASSES, bn_decay=bn_decay)
            MODEL.get_loss(pred, labels_pl, smpws_pl)
            losses = tf.get_collection('losses')
            total_loss = tf.add_n(losses, name='total_loss')
            tf.summary.scalar('total_loss', total_loss)
            for l in losses + [total_loss]:
                tf.summary.scalar(l.op.name, l)

            correct = tf.equal(tf.argmax(pred, 2), tf.to_int64(labels_pl)) # 1->2
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
            tf.summary.scalar('accuracy', accuracy)

            print "--- Get training operator"
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(total_loss, global_step=batch)
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
        
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init)

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'smpws_pl': smpws_pl, # added bc sem segm
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': total_loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'end_points': end_points}

        best_acc = -1
        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
             
            train_one_epoch(sess, ops, train_writer)
            eval_one_epoch(sess, ops, test_writer)

            # Save the variables to disk.
            if epoch % 1 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, MODEL_OUT_NAME))
                log_string("Model saved in file: %s" % save_path)


def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    log_string(str(datetime.now()))

    # Make sure batch data is of same size # or not since sem segm
    cur_batch_data = np.zeros((BATCH_SIZE,NUM_POINT,3)) # rm'd TRAIN_DATASET.num_channel()
    # cur_batch_label = np.zeros((BATCH_SIZE), dtype=np.int32) # seems to be for classification only
    cur_batch_label = np.zeros((BATCH_SIZE,NUM_POINT), dtype=np.int32) # for sem segm
    cur_batch_smpws = np.ones((BATCH_SIZE,NUM_POINT), dtype=np.float32) # for sem segm, don't know what other weights to use

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    batch_idx = 0
    #while TRAIN_DATASET.has_next_batch():
    num_batches_processed = 0 # for quicker analysis
    while num_batches_processed < 5:
        batch_data, batch_label = TRAIN_DATASET.next_batch(augment=True)
        #batch_data = provider.random_point_dropout(batch_data)
        bsize = batch_data.shape[0]
        cur_batch_data[0:bsize,...] = batch_data[:,:,:3] # either we keep PC as orig 6-features, or take only first 3
        # cur_batch_label[0:bsize] = batch_label # seems to be for classification only
        cur_batch_label[0:bsize,...] = batch_label # for sem segm
        # some broadcasting seems to have occurred

        # feed dictionary of placeholders that are indexed..
        feed_dict = {ops['pointclouds_pl']: cur_batch_data,
                     ops['labels_pl']: cur_batch_label,
                     ops['smpws_pl']: cur_batch_smpws, #TODO TBD
                     ops['is_training_pl']: is_training,}
        summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
            ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val[0:bsize] == batch_label[0:bsize])
        total_correct += correct
        total_seen += bsize
        loss_sum += loss_val
        if (batch_idx+1)%50 == 0:
            log_string(' ---- batch: %03d ----' % (batch_idx+1))
            log_string('mean loss: %f' % (loss_sum / 50))
            log_string('accuracy: %f' % (total_correct / float(total_seen)))
            total_correct = 0
            total_seen = 0
            loss_sum = 0
        batch_idx += 1

        num_batches_processed+=1

    TRAIN_DATASET.reset()
        
def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    is_training = False

    # Make sure batch data is of same size
    cur_batch_data = np.zeros((BATCH_SIZE,NUM_POINT,3)) # replaced TEST_DATASET.num_channel()
    # cur_batch_label = np.zeros((BATCH_SIZE), dtype=np.int32) # orig
    cur_batch_label = np.zeros((BATCH_SIZE, NUM_POINT), dtype=np.int32)
    cur_batch_smpws = np.ones((BATCH_SIZE,NUM_POINT), dtype=np.float32) # for sem segm, copying from train_one_epoch()


    total_correct = 0
    total_seen = 0
    loss_sum = 0
    batch_idx = 0
    shape_ious = []
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    
    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION ----'%(EPOCH_CNT))
    
    #while TEST_DATASET.has_next_batch():
    num_batches_processed = 0 # for quicker analysis
    while num_batches_processed < 5:
        batch_data, batch_label = TEST_DATASET.next_batch(augment=False)
        bsize = batch_data.shape[0]
        # for the last batch in the epoch, the bsize:end are from last batch
        #cur_batch_data[0:bsize,...] = batch_data # for clf
        cur_batch_data[0:bsize,...] = batch_data[:,:,:3] # for SS
        #cur_batch_label[0:bsize] = batch_label # classification probably
        cur_batch_label[0:bsize,...] = batch_label # for sem segm, matching to train_one_epoch()


        feed_dict = {ops['pointclouds_pl']: cur_batch_data,
                     ops['labels_pl']: cur_batch_label,
                     ops['smpws_pl']: cur_batch_smpws, #TODO TBD
                     ops['is_training_pl']: is_training}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
            ops['loss'], ops['pred']], feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        #pred_val = np.argmax(pred_val, 1)
        pred_val = np.argmax(pred_val, 2)
        correct = np.sum(pred_val[0:bsize] == batch_label[0:bsize]) # per point basis
        total_correct += correct
        # total_seen += bsize # orig. calculating acc on a per PC basis
        total_seen += bsize*NUM_POINT # calc'ing acc on a per point basis
        loss_sum += loss_val
        batch_idx += 1
        # for i in range(0, bsize): # per PC vector basis, but not v acc
        #     #l = batch_label[i]
        #     l = batch_label[i][0] # bc for SS, we currently have inefficient data structures
        #     total_seen_class[l] += 1
        #     #total_correct_class[l] += (pred_val[i] == l) # orig. suited for when pred_val is vector not matrix
        #     total_correct_class[l] += (Counter(pred_val[i]).most_common(1)[0][0] == l) # suited for pred_val as matrix. since cmp based on incomplete match

        for i in range(0, bsize):
            for j in range(NUM_POINT):
                l = cur_batch_label[i][j]
                total_seen_class[l] += 1
                total_correct_class += (pred_val[i][j] == l)

        num_batches_processed+=1
    
    log_string('eval mean loss: %f' % (loss_sum / float(batch_idx)))
    log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
    #log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float)))) #orig. not suited for vectors, would return nan
    log_string('eval avg class acc: %f' % np.mean([a/b if b!=0 else 0 for a,b in zip(total_correct_class, total_seen_class)])) # suited for vectors
    EPOCH_CNT += 1

    TEST_DATASET.reset()
    return total_correct/float(total_seen)


if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    log_string(RUN_DESCRIPTION)
    train()
    LOG_FOUT.close()
