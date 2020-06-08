# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Loads a sample video and classifies using a trained Kinetics checkpoint."""
#%%
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import sys
import os

# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"



import numpy as np
import matplotlib
import glob
# import setGPU
import tensorflow as tf

import i3d
# import skvideo
import utils.pre_process_rgb_flow as img_tool

import utils.kinetics_i3d_utils as ki3du


#%%
def model_fn(features, labels, mode, params):

    rgb_sample =features


    beta_0 = params.LAMBDA
    beta_1 = params.BETA_1
    beta_2 = params.BETA_2
    
    model_dir ='model/' #params.model_dir
    
    kinetics_classes = ki3du.load_kinetics_classes()
    
    if params.CYCLIC_ATTACK:
        cyclic_flag = 1.0
    else:
        cyclic_flag = 0.0
        
    if params.CYCLIC_PERTURBATION_ATTACK:
        cyclic_pert_flag = 1.0
    else:
        cyclic_pert_flag = 0.0
        
    adv_flag  = 1.0
    
    
    if ATTACK_CFG.FLICKERING_ATTACK:
        k_i3d = ki3du.kinetics_i3d(ckpt_path='',
                                   batch_size=None,
                                   init_model=False,
                                   rgb_input=rgb_sample,
                                   labels=labels,
                                   cyclic_flag_default_c=cyclic_flag,
                                   cyclic_pert_flag_default_c = cyclic_pert_flag,
                                   default_adv_flag_c=adv_flag)
    else: # "Sparse Adversarial Perturbations for Videos"  https://arxiv.org/pdf/1803.02536.pdf
        k_i3d = ki3du.kinetics_i3d_L12(ckpt_path='',
                                   batch_size=None,
                                   init_model=False,
                                   rgb_input=rgb_sample,
                                   labels=labels,
                                   cyclic_flag_default_c=cyclic_flag,
                                   default_adv_flag_c=adv_flag)


    inputs = k_i3d.rgb_input
    perturbation = k_i3d.eps_rgb
    adversarial_inputs_rgb = k_i3d.adversarial_inputs_rgb
    eps_rgb = k_i3d.eps_rgb
    adv_flag = k_i3d.adv_flag
    softmax = k_i3d.softmax
    softmax_clean = k_i3d.softmax_clean

    model_logits = k_i3d.model_logits
    # labels = k_i3d.labels
    # cyclic_input_flag = k_i3d.cyclic_flag
    # cyclic_pert_flag = k_i3d.cyclic_pert_flag
    norm_reg = k_i3d.norm_reg
    diff_norm_reg = k_i3d.diff_norm_reg
    laplacian_norm_reg = k_i3d.laplacian_norm_reg
    L12_loss = k_i3d.loss_L12
    thickness = k_i3d.thickness
    roughness = k_i3d.roughness
    thickness_relative= k_i3d.thickness_relative
    roughness_relative = k_i3d.roughness_relative


    predictions={#'prob': softmax,
                 'perturbation': tf.convert_to_tensor(perturbation)}

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,predictions=predictions) #,evaluation_hooks=[eval_summary_hook])



    if ATTACK_CFG.IMPROVE_ADV_LOSS:
        adversarial_loss = k_i3d.improve_adversarial_loss(margin=params.PROB_MARGIN,
                                                          targeted=params.TARGETED_ATTACK,
                                                          logits=params.USE_LOGITS)
    else:
        adversarial_loss = k_i3d.ce_adversarial_loss(targeted=params.TARGETED_ATTACK)


    if ATTACK_CFG.FLICKERING_ATTACK:
        regularizer_loss = beta_1 * norm_reg + beta_2 * diff_norm_reg + beta_2 * laplacian_norm_reg  # +lab_reg

    else:
        regularizer_loss = beta_1 * L12_loss
    
    
    weighted_regularizer_loss = beta_0 * regularizer_loss
    loss = adversarial_loss + weighted_regularizer_loss

    if mode == tf.estimator.ModeKeys.EVAL:
        prob_clean = softmax_clean

        if params.TARGETED_ATTACK:
            miss_cond = tf.argmax(softmax,axis=-1) ==  params.TARGETED_CLASS
        else:
            miss_cond = tf.argmax(softmax,axis=-1) != labels

        if params.TARGETED_ATTACK == False:
            valid_videos = tf.equal(tf.argmax(prob_clean,axis=-1) ,labels)
        else:
            valid_videos=None
            # Define the metrics:
        miss_cls_metric, miss_cls_metric_update_op =ki3du.miss_cls_fn(predictions=tf.argmax(softmax, axis=-1),
                                                                             labels=labels, 
                                                                             weights=valid_videos , #valid_videos,
                                                                             targeted=params.TARGETED_ATTACK)  # , weights=valid_videos)

        logging_hook = tf.train.LoggingTensorHook({"ACC": miss_cls_metric}, at_end=True)
        eval_metric_ops = {
        'ACC: 1- FOOLING_RATIO': (miss_cls_metric, miss_cls_metric_update_op)}
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=eval_metric_ops,prediction_hooks=[logging_hook]) #,evaluation_hooks=[eval_summary_hook])

    prob_to_min = k_i3d.to_min_prob
    prob_to_max = k_i3d.to_max_prob
    learning_rate_default = tf.constant(0.001, dtype=tf.float32)
    learning_rate = tf.placeholder_with_default(learning_rate_default, name='learning_rate',
                                                shape=learning_rate_default.shape)
    global_step = tf.train.get_or_create_global_step()
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    gradients = optimizer.compute_gradients(loss=loss, var_list=perturbation)
    train_op = optimizer.apply_gradients(gradients,global_step)


    logging_hook = tf.train.LoggingTensorHook({"loss": loss}, every_n_iter=10)

    tf.summary.scalar('Loss/total', loss)
    tf.summary.scalar('Loss/adversarial_loss', adversarial_loss)
    tf.summary.scalar('Loss/regularizer_loss', regularizer_loss)
    tf.summary.scalar('Loss/regularizer_loss_weighted', weighted_regularizer_loss)


    tf.summary.scalar('Loss/thickness', norm_reg)
    tf.summary.scalar('Loss/L12', L12_loss)
    tf.summary.scalar('Loss/first_order_temporal_diff', diff_norm_reg)
    tf.summary.scalar('Loss/second_order_temporal_diff', laplacian_norm_reg)


    tf.summary.scalar('Perturbation/thickness_%%', thickness_relative)
    tf.summary.scalar('Perturbation/roughness_%%', roughness_relative)
    tf.summary.scalar('Perturbation/max', tf.reduce_max(eps_rgb))
    tf.summary.scalar('Perturbation/min', tf.reduce_min(eps_rgb))
    
    tf.summary.scalar('Probability/prob_to_min', tf.reduce_mean(prob_to_min))
    tf.summary.scalar('Probability/prob_to_max', tf.reduce_mean(prob_to_max))

    summary_op = tf.summary.merge_all()

    train_summary_hook = tf.train.SummarySaverHook(
        save_steps=50,
        output_dir=os.path.join(model_dir, "train"),
        summary_op=summary_op)

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=[train_summary_hook,logging_hook])


#%%
def generate_input_fn(ATTACK_CFG, mode=tf.estimator.ModeKeys.EVAL, batch_size=1):




    tf_record_list_val=[]
    tf_record_list_train=[]
    for p in ATTACK_CFG.TF_RECORDS_VAL_PATH:
        tf_record_list_val += sorted(glob.glob(p + '/*.tfrecords'))
            
    for p in ATTACK_CFG.TF_RECORDS_TRAIN_PATH:
        tf_record_list_train += sorted(glob.glob(p + '/*.tfrecords'))
            
    tf_record_list_train = tf_record_list_train[:ATTACK_CFG.NUM_OF_TRAIN_TF_RECORDS]
    tf_record_list_val =tf_record_list_val[:ATTACK_CFG.NUM_OF_VAL_TF_RECORDS]        
    
    print("tf_record_list_train:")
    print('\n'.join(map(str, tf_record_list_train))) 

    
    print("tf_record_list_val:")
    print('\n'.join(map(str, tf_record_list_val))) 


    def _input_fn():

        if mode == tf.estimator.ModeKeys.TRAIN:
            tf_record_list = tf_record_list_train
        else:
            tf_record_list = tf_record_list_val

        dataset = tf.data.TFRecordDataset(filenames=tf_record_list, num_parallel_reads=os.cpu_count())
        if mode == tf.estimator.ModeKeys.TRAIN:
            dataset = dataset.repeat(1000)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.map(img_tool.parse_example_uint8, num_parallel_calls=os.cpu_count())
        dataset = dataset.prefetch(buffer_size=batch_size)

        if mode == tf.estimator.ModeKeys.EVAL:
            return dataset
        
        return dataset
        
        tf_record_list = [x.strip() for x in open(file_names)]
        dataset = tf.data.TFRecordDataset(filenames=tf_record_list, num_parallel_reads=os.cpu_count())

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        if is_training:
            buffer_size = batch_size * 2 + 1
            # dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=10000))
            # dataset = dataset.shuffle(buffer_size=buffer_size)

        # Transformation
        dataset = dataset.apply(tf.contrib.data.map_and_batch(img_tool.parse_example, batch_size=batch_size,num_parallel_calls=1,drop_remainder=True))

        # dataset = dataset.map(parse_record)
        # dataset = dataset.map(
        #     lambda image, label: (preprocess_image(image, is_training), label))
        #
        # dataset = dataset.repeat()
        # dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=1)

        # images, labels = dataset.make_one_shot_iterator().get_next()
        #
        # features = {'images': images}
        return dataset

    return _input_fn



#%%
kinetics_classes = ki3du.load_kinetics_classes()


if sys.argv.__len__()>1:
    cfg = ki3du.load_config(yml_path=sys.argv[1])
else:
    cfg = ki3du.load_config(yml_path='run_config.yml')


ATTACK_CFG = cfg.UNIVERSAL_ATTACK
source_class = ATTACK_CFG.TF_RECORDS_TRAIN_PATH[-1].split('/')[-2]
# assert source_class in kinetics_classes, "Oh no! {} not in kinetics classes".format(source_class)

# NUM_OF_VID_EACH_TF_RECORDS
# NUM_OF_TRAIN_TF_RECORDS
# NUM_OF_VAL_TF_RECORDS
if ATTACK_CFG.FLICKERING_ATTACK:
    attack_type='FLICKERING_ATTACK'
else:
    attack_type='SUP_ATTACK'
    
model_dir = os.path.join(ATTACK_CFG.PKL_RESULT_PATH,'{}/{}_t{}_v{}_'.format(attack_type,source_class,
                                                                         ATTACK_CFG.NUM_OF_VID_EACH_TF_RECORDS*ATTACK_CFG.NUM_OF_TRAIN_TF_RECORDS,
                                                                         ATTACK_CFG.NUM_OF_VID_EACH_TF_RECORDS*ATTACK_CFG.NUM_OF_VAL_TF_RECORDS))
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    
#%%

destribution = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
                                                #,devices=["/gpu:0", "/gpu:1"]) #(devices=["/gpu:0"])#(devices=["/gpu:0"]) #devices=["/gpu:0", "/gpu:1"]
run_config = tf.estimator.RunConfig(
               model_dir=model_dir,
               tf_random_seed=None,
               save_summary_steps=100,
               save_checkpoints_steps=100,
               #save_checkpoints_secs=_USE_DEFAULT,
               session_config=None,
               keep_checkpoint_max=5,
               #keep_checkpoint_every_n_hours=10000,
               log_step_count_steps=1,
               train_distribute=destribution,
               device_fn=None,
               protocol=None,
               eval_distribute=destribution,
               experimental_distribute=None,
               experimental_max_worker_delay_secs=None,
               session_creation_timeout_secs=7200
               )
#%%

checkpoint = tf.train.latest_checkpoint(checkpoint_dir=model_dir)

if checkpoint is None:
    regex = '^(?!.*global_step|.*eps).*$' #without global_step
    #regex = '.*' # all var include non-trainble
    checkpoint = str(cfg.MODEL.CKPT_PATH_WITH_ZERO_PERT)
    warm_start_from=tf.estimator.WarmStartSettings(
            ckpt_to_initialize_from=checkpoint,vars_to_warm_start=[regex])
    
    print("Begin new training, init from ckpt: {}".format(checkpoint))
    
else:
    warm_start_from=tf.estimator.WarmStartSettings(
            ckpt_to_initialize_from=checkpoint)
    print("Continue training, init from ckpt: {}".format(checkpoint))
# warm_start_from=None



estimator = tf.estimator.Estimator(
                model_fn=model_fn,
                model_dir=model_dir,
                config=run_config,
                params=ATTACK_CFG,
                warm_start_from=warm_start_from
                )


# log_dir="logs/profile/" + datetime.now().strftime("%Y%m%d-%H%M%S")

# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch = 3)

# profile_hook = tf.train.ProfilerHook(save_steps=50, output_dir='.')

train_spec = tf.estimator.TrainSpec(
    input_fn =generate_input_fn(ATTACK_CFG,mode=tf.estimator.ModeKeys.TRAIN,batch_size=8),
    max_steps=ATTACK_CFG.MAX_NUM_STEP,
    hooks=None
)
eval_spec = tf.estimator.EvalSpec(
    input_fn =generate_input_fn(ATTACK_CFG,mode=tf.estimator.ModeKeys.EVAL,batch_size=8),
    hooks=None, start_delay_secs=100
)

#%%

tf.estimator.train_and_evaluate(estimator= estimator,train_spec=train_spec,eval_spec=eval_spec)

#%%