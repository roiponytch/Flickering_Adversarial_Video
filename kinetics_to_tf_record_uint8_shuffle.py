import cv2
import numpy as np
import imageio
import tensorflow as tf
import i3d

import random
import skvideo.io
import sys
import os

import setGPU
from os import listdir
import glob
sys.path.insert(1, os.path.realpath(os.path.pardir))

import utils.pre_process_rgb_flow as img_tool
import utils.kinetics_i3d_utils as ki3du


def main(argv, arc):

    database_path = argv[1]
    tf_dst_folder = argv[2]

    # videos_base_path = argv[1]
    # class_name = argv[2]
    # tf_dst_folder = argv[3]
    # class_name = 'all'
    # database_path='/data/DL/Adversarial/ActivityNet/Crawler/Kinetics/database/val/'
    # tf_dst_folder='/home/ubadmin/pony/database/Kinetics/tfrecord_uint8/val/all_cls_shuffle/'
    
    num_videos_in_single_tfrecord=50
    
    # import pdb
    # pdb.set_trace()
    # videos_base_path = '/data/DL/Adversarial/ActivityNet/Crawler/Kinetics/database/val/'
    # class_name ='hula hooping'
    # tf_dst_folder = '/data/DL/Adversarial/ActivityNet/Crawler/Kinetics/database/tfrecord_uint8/val/'
    
    
    
    if not os.path.exists(tf_dst_folder):
        os.makedirs(tf_dst_folder)
    
    n_frames = ki3du._SAMPLE_VIDEO_FRAMES
    
    kinetics_classes =ki3du.load_kinetics_classes()
    
    videos_list_path=glob.glob(database_path + '*/*.mp4')
    random.shuffle(videos_list_path)
           
        
    k=0
    i=0
#%%
    for v in videos_list_path:
        if i%num_videos_in_single_tfrecord ==0:
            if k>0:
                writer.close()
                
            target_folder =os.path.join(tf_dst_folder)
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)
                
            train_filename = os.path.join(target_folder,'kinetics_N_{}_{:04}.tfrecords'.format(num_videos_in_single_tfrecord, k) ) # address to save the TFRecords file
            # open the TFRecords file
            writer = tf.python_io.TFRecordWriter(train_filename)
            k+=1
    
        cls =  v.split('/')[-2]
        cls_id = kinetics_classes.index(cls)
        vid_path = v
        
        
        if os.path.exists(v)==False:
            continue
        try:
            frames = skvideo.io.vread(vid_path)
            # frames = frames.astype('float32') / 128. - 1
        except:
            os.remove(vid_path)
            continue
            
        if frames.shape[0] < ki3du._SAMPLE_VIDEO_FRAMES:
            continue
            #frames = np.pad(frames, ((0, _SAMPLE_VIDEO_FRAMES-frames.shape[0]),(0,0),(0,0),(0,0)),'wrap')
        else:
            frames=frames[-ki3du._SAMPLE_VIDEO_FRAMES:]
            
        # prob = sess.run(softmax,feed_dict={rgb_input:frames})
        # top_id = prob.argmax()
        # if cls_id!=top_id:
        #     continue
        feature = {'train/label': img_tool._int64_feature(cls_id),
                   'train/video': img_tool._bytes_feature(frames.tobytes())}
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
        i+=1
#%%
    writer.close()


if __name__ == '__main__':
    main(sys.argv, len(sys.argv))