##
import sys
sys.path.append(".")
import time
import os
import numpy as np
import random

import torch
import torch.cuda as cuda
from torch.autograd import Variable
import torch.nn as nn
import torchvision
import requests
import decord
# import setGPU

from utils_cv.action_recognition.data import KINETICS, Urls
from utils_cv.action_recognition.model import VideoLearner, VideoLearnerAdversarial, Adversarial_metrics
from utils_cv.action_recognition.dataset import VideoRecord, VideoDataset, get_transforms, get_spatial_transforms, get_normalize_transforms
from utils_cv.action_recognition.data import Urls
from utils_cv.common.gpu import system_info, torch_device, num_devices
from utils_cv.common.data import data_path, download
import glob
LABELS =KINETICS.class_names
# from torchvision.models.video.resnet import VideoResNet, BasicBlock, R2Plus1dStem, Conv2Plus1D
system_info()

## TRIA
if num_devices()>1:
    DEVICES_IDS=[0,1,2,3]
else:
    DEVICES_IDS=[0]

# Number of training epochs
EPOCHS = 22

# Learning rate
LR = 0.001
MIXED_PREC= False

LAMBDA = 1.0 #0.01#10
BETA_1 =0.5

L_INF_PERT_NORM = 0.1
BASE_MODEL = "mc3_18" #"mc3_18", "r2plus1d_18", "r3d_18"

USE_LOGITS =False
IMPROVE_LOSS =True

CYCLIC_PERT = False
ATTACK_TYPE="flickering"  #"flickering", "L12"

if ATTACK_TYPE== 'L12':
    USE_LOGITS =False
    IMPROVE_LOSS =True

TARGETED_ATTACK =False

INIT_PERT_FROM_LAST_CKPT=True
CONTINUE_TRAIN =True

KINETICS_DB_PATH='/data/DL/Adversarial/ActivityNet/Crawler/Kinetics/database/'
computervision_recipes_path=os.getcwd()

kinetics_train_dataset_path = KINETICS_DB_PATH+'/train/'
kinetics_val_dataset_path = KINETICS_DB_PATH+'/val/'
kinetics_test_dataset_path = KINETICS_DB_PATH +'/test/'

train_path = kinetics_val_dataset_path
val_path = kinetics_test_dataset_path

path_id_list =[]

train_split_path= 'data/kinetics400/train_split.txt'
val_split_path= 'data/kinetics400/val_split.txt'

NUM_OF_SAMPLES_TRAIN = 15000 #50, 100, 200, 500, 1050, 5000, 15000
NUM_OF_SAMPLES_VAL = 2000
NUM_REPEAT = int(NUM_OF_SAMPLES_VAL/NUM_OF_SAMPLES_TRAIN)
random.seed(a=13)

with open(train_split_path, 'w') as filehandle:
    all_train_videos_path = glob.glob(os.path.join(train_path,'*/','*.mp4'))
    all_train_videos_path.sort()
    random.shuffle(all_train_videos_path)

    if NUM_REPEAT >1:
        video_list_for_train = all_train_videos_path[:NUM_OF_SAMPLES_TRAIN]*NUM_REPEAT
        random.shuffle(video_list_for_train)
    else:
        video_list_for_train =  all_train_videos_path[:NUM_OF_SAMPLES_TRAIN]
    for p in video_list_for_train:
    
        class_id = LABELS.index(p.split('/')[-2])
        entry ='{},{}'.format(p.replace('.mp4',''),class_id)
        
        filehandle.write('%s\n' % entry)
    
with open(val_split_path, 'w') as filehandle:
    all_val_videos_path =glob.glob(os.path.join(val_path,'*/','*.mp4'))
    all_val_videos_path.sort()
    random.shuffle(all_val_videos_path)

    for p in all_val_videos_path[:NUM_OF_SAMPLES_VAL]:

        class_id = LABELS.index(p.split('/')[-2])
        entry ='{},{}'.format(p.replace('.mp4',''),class_id)
        
        filehandle.write('%s\n' % entry)
##
IM_SCALE = 128  # resize then crop
INPUT_SIZE = 112  # input clip size: 3 x NUM_FRAMES x 112 x 112

# video sample to download
sample_video_url = Urls.webcam_vid

# file path to save video sample
video_fpath = data_path() / "sample_video.mp4"

# prediction score threshold
SCORE_THRESHOLD = 0.01

# Averaging 5 latest clips to make video-level prediction (or smoothing)
AVERAGING_SIZE = 5  

##


if BASE_MODEL =="kinetics":
    BATCH_SIZE_ARRAY = [8,1,4,5,6]
    MODEL_INPUT_SIZE = 32
elif  BASE_MODEL =="r2plus1d_18":
    BATCH_SIZE_ARRAY = [8, 16, 16, 20, 20]
    MODEL_INPUT_SIZE = 16
elif  BASE_MODEL =="mc3_18":
    BATCH_SIZE_ARRAY = [8, 20, 16, 20, 20]
    MODEL_INPUT_SIZE = 16
elif  BASE_MODEL =="r3d_18":
    BATCH_SIZE_ARRAY = [8, 16, 16, 20, 20]
    MODEL_INPUT_SIZE = 16
else:
    print("Unknown base model")
    assert False
# Number of consecutive frames used as input to the DNN. Recommended: 32 for high accuracy, 8 for inference speed.


# Batch size. Reduce if running out of memory.
BATCH_SIZE = BATCH_SIZE_ARRAY[len(DEVICES_IDS)]


##

kinetics_dataset_path = ''
data = VideoDataset(kinetics_dataset_path,
                    seed= None,
                    train_pct= 0.75,
                    num_samples= 1,
                    sample_length=MODEL_INPUT_SIZE,
                    sample_step= 1,
                    temporal_jitter = False,
                    temporal_jitter_step= 2,
                    random_shift = False,
                    batch_size = BATCH_SIZE,
                    warning = False,
                    train_split_file= train_split_path,
                    test_split_file = val_split_path,
                    video_ext="mp4",
                    train_transforms= get_transforms(train=False),
                    test_transforms= get_transforms(train=False)
)

learner= VideoLearnerAdversarial(data,
                        base_model = BASE_MODEL,
                        sample_length=MODEL_INPUT_SIZE,
                        l_inf_pert_norm=L_INF_PERT_NORM,
                        attack_type=ATTACK_TYPE,
                        cyclic_pert=CYCLIC_PERT,
                        labaels_id_to_text=LABELS)



##

DEST_FOLDER=computervision_recipes_path+'/results/{}/generalization/universal/val_test/all_cls_shuffle_{}/t_{}_v_{}_linf_{}_lambda_{}_beta1_{}_/'.format(
                      learner.model_name,
                      ATTACK_TYPE,
                      NUM_OF_SAMPLES_TRAIN,
                      NUM_OF_SAMPLES_VAL,
                      L_INF_PERT_NORM,
                      LAMBDA,
                      BETA_1)

## load pert from last ckpt
start_epoch =1

if INIT_PERT_FROM_LAST_CKPT:
    init_from_path = glob.glob(DEST_FOLDER + '/*.npy')
    if len(init_from_path) >0:
        init_from_path = glob.glob(DEST_FOLDER + '/*.npy')
        init_from_path.sort(key=os.path.getmtime)
        pert = np.load(init_from_path[-1], allow_pickle=True)[-1]['valid/perturbation']
        learner.pert_model.init_perturbation(pert)
        print("Success! init from last ckpt")
    else:
        print("Failed! to init from last ckpt. init with default")

if CONTINUE_TRAIN:
    init_from_path = glob.glob(DEST_FOLDER + '/*.npy')
    if len(init_from_path) >0:
        init_from_path = glob.glob(DEST_FOLDER + '/*.npy')
        init_from_path.sort(key=os.path.getmtime)
        start_epoch = int(init_from_path[-1].split('_')[-1].split('.')[0]) +1
        print("Success! to continue from last epoch. init with {}".format(start_epoch))
    else:
        print("Failed! to continue from last epoch. init with 1")


loss_params_dict={'lambda_': LAMBDA,
                  'beta_1': BETA_1,
                  'targeted_attack': TARGETED_ATTACK,
                  'target_class_id': None,
                  'target_class_name': None,
                  'improve_loss': IMPROVE_LOSS,
                  'use_logits': USE_LOGITS}
learner.fit(lr=LR,
            epochs=EPOCHS,
            loss_params_dict=loss_params_dict,
            devices_ids=DEVICES_IDS,
            mixed_prec=MIXED_PREC,
            model_dir=DEST_FOLDER,
            save_model=True,
            model_name=learner.model_name,
            start_epoch=start_epoch)


