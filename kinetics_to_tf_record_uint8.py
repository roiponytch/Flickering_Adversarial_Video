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

    videos_base_path = argv[1]
    class_name = argv[2]
    tf_dst_folder = argv[3]
    
    # videos_base_path = '/data/DL/Adversarial/ActivityNet/Crawler/Kinetics/database/val/'
    # class_name ='hula hooping'
    # tf_dst_folder = '/data/DL/Adversarial/ActivityNet/Crawler/Kinetics/database/tfrecord_uint8/val/'
    
    
    
    if class_name=='all':
        classes_list =listdir(videos_base_path)
    else:
        classes_list =[class_name]
    
    
    if not os.path.exists(tf_dst_folder):
        os.makedirs(tf_dst_folder)
    
    n_frames = ki3du._SAMPLE_VIDEO_FRAMES
    
    kinetics_classes =ki3du.load_kinetics_classes()
    
    
    for c in classes_list:
        videos_list = os.path.join(videos_base_path,c)
        if not os.path.exists(videos_list):
            print('{} not exist'.format(videos_list))
            continue
        
        video_list_path=glob.glob(videos_list + '*/*.mp4')
        k=0
        for i,v in enumerate(video_list_path):
            if i%100 ==0:
                if i>0:
                    writer.close()
                    k+=1
                target_folder =os.path.join(tf_dst_folder,c)
                if not os.path.exists(target_folder):
                    os.makedirs(target_folder)
                    
                train_filename = os.path.join(target_folder,'kinetics_{}_{:04}.tfrecords'.format(c, k) ) # address to save the TFRecords file
                # open the TFRecords file
                writer = tf.python_io.TFRecordWriter(train_filename)
        
            cls =  c
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
    
        writer.close()


if __name__ == '__main__':
    main(sys.argv, len(sys.argv))