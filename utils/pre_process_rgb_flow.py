import cv2
import numpy as np
import imageio
import tensorflow as tf
import os
import sys
import random
import utils.kinetics_i3d_utils as kutils
# import setGPU
import skvideo.io

def run_npy(vid_npy):
    vid = vid_npy.squeeze()
    num_frames= vid.shape[0]
    idx=0
    while True:
        frame =cv2.cvtColor(((vid[idx,...]+1.0)*127.5).astype(np.uint8),cv2.COLOR_RGB2BGR)
        cv2.imshow('video_npy', frame,)
        c = cv2.waitKey(40) & 0xFF
        if c == 27 or c == ord('q'):
            break
        idx = (idx + 1) % num_frames
    cv2.destroyAllWindows()

def run_mp4(path):
    frames = skvideo.io.vread(path)
    frames = frames.astype('float32') / 128. - 1
    run_npy(frames)

def image_resize(image, width = None, height = None, inter = cv2.INTER_LINEAR):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]


    r = max(float(width) / w, float(height) / h)
    dim = (int(w * r), int(h * r))


    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def crop_center_image(image, rec_len):
    # crop center
    (h, w) = image.shape[:2]
    x_1 = (w - rec_len) // 2
    x_2 = (h - rec_len) // 2
    cropped = image[x_2:x_2+rec_len, x_1:x_1+rec_len]
    return cropped



def video_to_image_and_of(video_path, target_fps=25, resize_height=256, crop_size=224, n_steps=100,plotting=False,flow=False):
# preprocessing:


    clip_frames = []
    clip_frames_flow = []
    bit = 0
    capture = cv2.VideoCapture(video_path)
    fps = capture.get(cv2.CAP_PROP_FPS)

    frame_gap = int(round(fps / target_fps))
    frame_num = 1

    ret, frame1 = capture.read()
    resized_frame1 = image_resize(frame1, height = resize_height,width=resize_height)
    prvs = cv2.cvtColor(resized_frame1,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255

    # extract features
    while (capture.isOpened()) & (bit == 0):
        flag, frame = capture.read()
        if flag == 0:
            bit = 1
            print("******ERROR: Could not read frame in " + video_path + " frame_num: " + str(frame_num))
            break

        #name = params['res_vids_path'] + str(frame_num) + 'frame.jpg'
        #cv2.imwrite(name, frame)
        #cv2.imshow("Vid", frame)
        #key_pressed = cv2.waitKey(10)  # Escape to exit

        # process frame (according to the correct frame rate)
        if frame_num % frame_gap == 0:
            # RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized_frame = image_resize(image,width=resize_height, height = resize_height)
            res = resized_frame.astype('float32') / 128. - 1
            res = crop_center_image(res, crop_size)
            clip_frames.append(res)

            if plotting:
                res_to_plot = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
                res_to_plot = res_to_plot + 1.0 / 2.0
                cv2.imshow("Vid", res_to_plot)
                key_pressed = cv2.waitKey(10)  # Escape to exit

            # FLOW
            if flow:
                image_flow = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2GRAY)
                # flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                optical_flow = cv2.DualTVL1OpticalFlow_create()
                flow = optical_flow.calc(prvs, image_flow, None)
                flow[flow > 20] = 20
                flow[flow < -20] = -20
                flow = flow / 20.
                flow = crop_center_image(flow, crop_size)
                clip_frames_flow.append(flow)

                # potting:
                if plotting:
                    flow_temp = (flow + 1.0) / 2.0
                    last_channel = np.zeros((crop_size,crop_size), dtype=float) + 0.5
                    flow_to_plot = np.dstack((flow_temp, last_channel))
                    cv2.imshow("Vid-flow", flow_to_plot)
                    key_pressed = cv2.waitKey(10)  # Escape to exit


                prvs = image_flow

        frame_num += 1

    capture.release()


     # (int(round(frame_num / frame_gap)) < n_steps)
    if frame_num>=n_steps:
        frames = np.array(clip_frames)[-n_steps:]
        frames_flow = np.array(clip_frames_flow)
        frames = np.expand_dims(frames, axis=0)
        frames_flow = np.expand_dims(frames_flow, axis=0)
    else:
        frames = np.array(clip_frames)
        frames_flow = np.array(clip_frames_flow)
        frames = np.expand_dims(frames, axis=0)
        frames_flow = np.expand_dims(frames_flow, axis=0)



    return frames, frames_flow


def frames_to_gif(frames, output_path, fps=25):
    imageio.mimsave(output_path, frames, fps=fps)

# np.save('rgb.npy', frames)
# np.save('flow.npy', frames_flow)

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def parse_example(serialized):
    context_features = {'train/label' : tf.FixedLenFeature((), tf.int64),
                        'train/video' : tf.VarLenFeature(dtype=tf.float32)}
    # context_features = {'train/label' : tf.FixedLenFeature((), tf.int64),
    #                     'train/video' : tf.FixedLenFeature((), tf.string)}
    # import pdb 
    # pdb.set_trace()
    # context_parsed,_ = tf.parse_single_sequence_example(serialized=serialized,context_features =context_features,sequence_features={})
    context_parsed = tf.io.parse_single_example(serialized=serialized,features=context_features)

    # video =  tf.image.decode_jpeg(context_parsed['train/video'], channels=3)
    videos_raw = context_parsed['train/video']
   
    
    
    # video =  tf.reshape(tf.sparse.to_dense(videos_raw), shape=[ serialized.shape[0].value,-1,224,224,3])
    # label = context_parsed['train/label']
    video =  tf.reshape(tf.sparse.to_dense(context_parsed['train/video']), shape=[-1,224,224,3])
    label = context_parsed['train/label']

    return video, label

def parse_example_batch(serialized):
    context_features = {'train/label' : tf.FixedLenFeature((), tf.int64),
                        'train/video' : tf.VarLenFeature(dtype=tf.float32)}
    # context_features = {'train/label' : tf.FixedLenFeature((), tf.int64),
    #                     'train/video' : tf.FixedLenFeature((), tf.string)}
    # import pdb 
    # pdb.set_trace()
    # context_parsed,_ = tf.parse_single_sequence_example(serialized=serialized,context_features =context_features,sequence_features={})
    # context_parsed = tf.io.parse_single_example(serialized=serialized,features=context_features)
    context_parsed = tf.io.parse_example(serialized=serialized,features=context_features)


    # video =  tf.image.decode_jpeg(context_parsed['train/video'], channels=3)
    videos_raw = context_parsed['train/video']
   
    # videos_raw = tf.decode_raw(context_parsed['train/video'], tf.uint8)
    

    # video =  tf.reshape(tf.sparse.to_dense(videos_raw), shape=[ serialized.shape[0].value,-1,224,224,3])
    # label = context_parsed['train/label']
    video =  tf.reshape(tf.sparse.to_dense(videos_raw), shape=[serialized.shape[0].value,-1,224,224,3])
    label = context_parsed['train/label']
    
    

    return video, label


def parse_example_uint8(serialized):
    context_features = {'train/label' : tf.FixedLenFeature((), tf.int64),
                        'train/video' : tf.FixedLenFeature([],tf.string)}
    # context_features = {'train/label' : tf.FixedLenFeature((), tf.int64),
    #                     'train/video' : tf.FixedLenFeature((), tf.string)}
    # import pdb 
    # pdb.set_trace()
    # context_parsed,_ = tf.parse_single_sequence_example(serialized=serialized,context_features =context_features,sequence_features={})
    # context_parsed = tf.io.parse_single_example(serialized=serialized,features=context_features)
    context_parsed = tf.io.parse_example(serialized=serialized,features=context_features)


    # video =  tf.image.decode_jpeg(context_parsed['train/video'], channels=3)
    videos_raw = context_parsed['train/video']
   
    videos_raw = tf.decode_raw(context_parsed['train/video'], tf.uint8)
    

    # video =  tf.reshape(tf.sparse.to_dense(videos_raw), shape=[ serialized.shape[0].value,-1,224,224,3])
    # label = context_parsed['train/label']
    video =  tf.reshape(videos_raw, shape=[serialized.shape[0].value,-1,224,224,3])
    label = context_parsed['train/label']
    
    video = tf.cast(video,dtype=tf.float32) / 128. - 1

    return video, label


def random_videos(videos_folder,n_steps, num_of_vid,dest_folder, model):

    sub_folders = os.listdir(videos_folder)
    random.shuffle(sub_folders)
    sub_folders = sub_folders[:num_of_vid]

    for cls in sub_folders:
        cls_folders = os.path.join(videos_folder,cls)
        cls_videos_list = os.listdir(cls_folders)
        random.shuffle(cls_videos_list)
        vid_name = cls_videos_list[0]
        full_vid_path = os.path.join(cls_folders,vid_name)
        rgb, of, = video_to_image_and_of(full_vid_path, n_steps=n_steps)
        prob = model(rgb)
        top_id = prob.argmax()
        if model.get_kinetics_classes().index(cls.replace('_',' '))!=top_id:
            continue
        dest_video_name = os.path.join(dest_folder,"rgb_{}@{}.npy".format(vid_name.split('.')[0],cls))
        np.save(dest_video_name,rgb)


def main(unused_argv):
    model = kutils.kinetics_i3d_inference()
    random_videos(videos_folder='/data/DL/Adversarial/kinetics-downloader/dataset/valid/',
                  n_steps=90,
                  num_of_vid=40,
                  dest_folder='/data/DL/Adversarial/kinetics-i3d/data/videos_for_tests/npy/'
                  ,model=model)
    
    
def main1(unused_argv):


    f, of, = video_to_image_and_of('/data/DL/Adversarial/kinetics-downloader/dataset/train/juggling_balls/7pxmupXYnuo.mp4',n_steps=90)

    video_base_path = '/media/ROIPO/Data/projects/Adversarial/database/UCF-101/video/'
    video_list_path = ['/media/ROIPO/Data/projects/Adversarial/database/UCF-101/ucfTrainTestlist/testlist01.txt',
                       '/media/ROIPO/Data/projects/Adversarial/database/UCF-101/ucfTrainTestlist/testlist02.txt',
                       '/media/ROIPO/Data/projects/Adversarial/database/UCF-101/ucfTrainTestlist/testlist03.txt']
    label_map_path='/media/ROIPO/Data/projects/Adversarial/kinetics-i3d/data/label_map_ucf_101.txt'

    class_target = ['triple jump']

    n_frames = 90

    ucf_classes = [x.strip() for x in open(label_map_path)]

    for vl in video_list_path:

        video_list = [x.strip() for x in open(vl)]
        video_list=  [v for v in video_list if any(c in v for c in class_target)]
        testListName = vl.split('/')[-1].split('.')[0]
        train_filename = 'data/{}_{}.tfrecords'.format(testListName,'_'.join(class_target))  # address to save the TFRecords file
        # open the TFRecords file
        writer = tf.python_io.TFRecordWriter(train_filename)

        for i, v in enumerate(video_list):
            cls, vid =  v.split('/')
            cls_id = ucf_classes.index(cls)
            vid_path = video_base_path + vid
            frames, flow_frames = video_to_image_and_of(video_path=vid_path, n_steps=n_frames)
            if frames.shape[1] < n_frames-1:
                continue
            feature = {'train/label': _int64_feature(cls_id),
                       'train/video': _float_list_feature(frames.reshape([-1]))}
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

        writer.close()

        # video_path= ['/media/ROIPO/Data/projects/Adversarial/database/UCF-101/video/v_BreastStroke_g01_c01.avi',
        # '/media/ROIPO/Data/projects/Adversarial/database/UCF-101/video/v_BreastStroke_g01_c02.avi']
        #



    # for i, vp in enumerate(video_path):
    #     frames, flow_frames = video_to_image_and_of(video_path=vp,n_steps=80)
    #     feature = {'train/label': _int64_feature(1),
    #                'train/video': _float_list_feature(frames.reshape([-1]))}
    #     example = tf.train.Example(features=tf.train.Features(feature=feature))
    #     writer.write(example.SerializeToString())
    

    # sys.stdout.flush()
    
    #     frames_list.append(frames)
    # np.save('data/db.npy', frames_list)

if __name__ == '__main__':
  tf.app.run(main)