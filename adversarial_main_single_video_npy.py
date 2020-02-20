
#%%
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import numpy as np
import setGPU
import matplotlib
import matplotlib.pyplot as plt
import imageio
import pickle
import tensorflow as tf
sys.path.insert(1, os.path.realpath(os.path.pardir))
import utils.kinetics_i3d_utils as ki3du
import i3d
# import skvideo
import utils.pre_process_rgb_flow as img_tool

#%%

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

tf_config = tf.ConfigProto()
# tf_config.gpu_options.per_process_gpu_memory_fraction = 0.99

eval_type = 'rgb'
cfg = ki3du.load_config(yml_path='run_config.yml')
ATTACK_CFG = cfg.SINGLE_VIDEO_ATTACK

kinetics_classes = ki3du.load_kinetics_classes(eval_type)

k_i3d = ki3du.kinetics_i3d(ckpt_path=cfg.MODEL.CKPT_PATH, batch_size=ATTACK_CFG.BATCH_SIZE)

if ATTACK_CFG.IMPROVE_ADV_LOSS:
    adversarial_loss = k_i3d.improve_adversarial_loss(margin=ATTACK_CFG.PROB_MARGIN,
                                                      targeted = ATTACK_CFG.TARGETED_ATTACK,
                                                      logits = ATTACK_CFG.USE_LOGITS)
else:
    adversarial_loss = k_i3d.ce_adversarial_loss(targeted=ATTACK_CFG.TARGETED_ATTACK)

beta_0_default = tf.constant(1, dtype=tf.float32)
beta_0 = tf.placeholder_with_default(beta_0_default, name='beta_0', shape=beta_0_default.shape)

beta_1_default = tf.constant(0.1, dtype=tf.float32)
beta_1 = tf.placeholder_with_default(beta_1_default, name='beta_1', shape=beta_1_default.shape)

beta_2_default = tf.constant(0.1, dtype=tf.float32)
beta_2 = tf.placeholder_with_default(beta_2_default, name='beta_2', shape=beta_2_default.shape)

beta_3_default = tf.constant(0.1, dtype=tf.float32)
beta_3 = tf.placeholder_with_default(beta_3_default, name='beta_3', shape=beta_3_default.shape)

regularizer_loss = beta_1*k_i3d.norm_reg + beta_2*k_i3d.diff_norm_reg +beta_3*k_i3d.laplacian_norm_reg # +lab_reg
weighted_regularizer_loss = beta_0 * regularizer_loss

loss = adversarial_loss + weighted_regularizer_loss

inputs = k_i3d.rgb_input
perturbation = k_i3d.eps_rgb
adversarial_inputs_rgb = k_i3d.adversarial_inputs_rgb
eps_rgb =  k_i3d.eps_rgb
adv_flag = k_i3d.adv_flag
softmax  = k_i3d.softmax
model_logits = k_i3d.model_logits
labels = k_i3d.labels
cyclic_flag = k_i3d.cyclic_flag
norm_reg = k_i3d.norm_reg
diff_norm_reg = k_i3d.diff_norm_reg
laplacian_norm_reg = k_i3d.laplacian_norm_reg
thickness = k_i3d.thickness
roughness = k_i3d.roughness
prob_to_min = k_i3d.to_min_prob
prob_to_max = k_i3d.to_max_prob
sess = k_i3d.sess

learning_rate_default = tf.constant(0.001, dtype=tf.float32)
learning_rate = tf.placeholder_with_default(learning_rate_default, name='learning_rate', shape=learning_rate_default.shape)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
gradients = optimizer.compute_gradients(loss=loss, var_list=perturbation)
train_op = optimizer.apply_gradients(gradients)
sess.run(tf.variables_initializer(optimizer.variables()))


#%%

_cyclic_flag = np.float32(ATTACK_CFG.CYCLIC_ATTACK)
_adv_flag =1.0

_lr=0.001

# regularization:
_beta_0 = ATTACK_CFG.LAMBDA
_beta_1 = ATTACK_CFG.BETA_1
_beta_2 = ATTACK_CFG.BETA_2
_beta_3 = ATTACK_CFG.BETA_2


npy_path = ATTACK_CFG.NPY_PATH
if not os.path.exists(npy_path):
    print('npy\'s path not exist!')

result_path = ATTACK_CFG.PKL_RESULT_PATH
if not os.path.exists(result_path):
    os.makedirs(result_path)

save_result = True
stats = []

#%%
k=0

for video in os.listdir(npy_path):

    k+=1
    
    res_dict={}
    video_path = os.path.join(npy_path,video)
    rgb_sample =np.load(video_path)[0,   -ki3du._SAMPLE_VIDEO_FRAMES:]
    rgb_sample = rgb_sample[np.newaxis]
    correct_cls = video.split('@')[-1].split('.')[0].replace('_',' ')
    sample_label =[kinetics_classes.index(correct_cls)]
    
    feed_dict_for_clean_eval = {inputs: rgb_sample, adv_flag: 0}

    model_softmax = sess.run(softmax, feed_dict=feed_dict_for_clean_eval)

    sample_label = sample_label[0]
    top_id = model_softmax.argmax()
    top_id_prob = model_softmax.max()
    gt_label_name = kinetics_classes[sample_label]
    predict_label_name = kinetics_classes[top_id]


    if top_id!=sample_label:
        print('skip video: source class {}, predict class {}, prob: {:.2f}'.format(gt_label_name,predict_label_name,top_id_prob))
        continue
              
    # if save_result:
    #     dict_result_path = os.path.join(result_path, 'res_beta_1_{}_{}_{:05d}'.format(_beta_1,gt_label_name.replace(' ','_'),k) + '.pkl')
        

    # if os.path.exists(dict_result_path):
    #     print('{} exist'.format(dict_result_path)
    #     continue
    
    
    res_dict = {}
    
    correct_cls_id= sample_label
    correct_cls = kinetics_classes[correct_cls_id]
    
    if ATTACK_CFG.TARGETED_ATTACK:
        target_class = ATTACK_CFG.TARGETED_CLASS 
        is_adversarial = tf.cast(tf.reduce_all(tf.equal(tf.argmax(softmax,axis=-1),labels)), dtype=tf.float32)

    else:
        target_class = correct_cls
        is_adversarial = tf.cast(tf.reduce_all(tf.not_equal(tf.argmax(softmax,axis=-1),labels)), dtype=tf.float32)

        
    target_class_id = kinetics_classes.index(target_class)
    

    feed_dict_for_train = {inputs: rgb_sample,
                           labels: [target_class_id],
                           cyclic_flag: _cyclic_flag,
                           adv_flag: _adv_flag,
                           learning_rate:_lr,
                           beta_0: _beta_0,
                           beta_1: _beta_1,
                           beta_2: _beta_2,
                           beta_3: _beta_3}

    res_dict['correct_cls_prob']=top_id_prob
    res_dict['correct_cls'] = correct_cls
    res_dict['correct_cls_id'] = correct_cls_id
    res_dict['softmax_init'] =model_softmax
    res_dict['rgb_sample'] = rgb_sample


    total_loss_l=[]
    adv_loss_l = []
    reg_loss_l= []
    norm_reg_loss_l=[]
    diff_norm_reg_loss_l=[]
    laplacian_norm_reg_l = []
    roughness_l = []
    thickness_l=[]

    _model_softmax=[]
    _perturbation=[]
    
    correct_cls_prob_l=[]
    prob_to_max_l=[]
    max_prob_l=[]
    prob_to_min_l=[]


    step = 0
    max_step =ATTACK_CFG.MAX_NUM_STEP

    sess.run(eps_rgb.initializer)
    sess.run(tf.variables_initializer(optimizer.variables()))

    fig = plt.figure()

#%%
    while True:  # to repeat with decreased epsilons if necessary

        _, _total_loss, _adv_loss, _reg_loss, _norm_reg_loss, _diff_norm_reg_loss, _laplacian_norm_reg, _thickness, _roughness, _prob_to_max, _prob_to_min = \
        sess.run(fetches=[train_op, loss, adversarial_loss, regularizer_loss, norm_reg, diff_norm_reg, laplacian_norm_reg, thickness, roughness,prob_to_max,prob_to_min ],
        feed_dict=feed_dict_for_train)

        _is_adversarial ,curr_model_softmax =sess.run([is_adversarial,softmax], feed_dict=feed_dict_for_train)
        _is_adversarial=_is_adversarial.astype(np.bool)
        _model_softmax.append(curr_model_softmax)
        
        correct_cls_prob_l.append( curr_model_softmax[0, correct_cls_id])
        max_prob_l.append(curr_model_softmax[0].max())
        prob_to_max_l.append(_prob_to_max[0])
        prob_to_min_l.append(_prob_to_min[0])

        print(
            "Step: {:05d}, Total Loss: {:.5f}, Cls Loss: {:.5f}, Total Reg Loss: {:.5f}, Fat Loss: {:.5f}, Diff Loss: {:.5f}, prob_correct_cls: {:.5f}"
            ", top_prob: {:.5f}, prob_to_min: {:.6f}, prob_to_max: {:.6f}, thickness: {:.5f} ({:.2f} %), roughness: {:.5f} ({:.2f} %)".format(step, _total_loss,
                                                                                                    _adv_loss,
                                                                                                    _reg_loss,
                                                                                                    _norm_reg_loss,
                                                                                                    _diff_norm_reg_loss,
                                                                                                    curr_model_softmax[
                                                                                                        0, correct_cls_id],
                                                                                                    curr_model_softmax[
                                                                                                        0].max(),
                                                                                                    _prob_to_min[0],
                                                                                                    _prob_to_max[0],
                                                                                                    _thickness,
                                                                                                    _thickness / 2.0 * 100,
                                                                                                    _roughness,
                                                                                                   _roughness / 2.0 * 100))


        total_loss_l.append(_total_loss)
        adv_loss_l.append(_adv_loss)
        reg_loss_l.append(_reg_loss)
        norm_reg_loss_l.append(_norm_reg_loss)
        diff_norm_reg_loss_l.append(_diff_norm_reg_loss)
        laplacian_norm_reg_l.append(_laplacian_norm_reg)
        thickness_l.append(_thickness / 2.0 * 100)
        roughness_l.append(_roughness / 2.0 * 100)
                

        
        if np.random.rand() > 0.9:
            plt.clf()
            ax1=fig.add_subplot(4,1,1)
            ax1.semilogy(total_loss_l,'r',label='total_loss')
            ax1.semilogy(adv_loss_l,'--b',label='adv_loss')
            ax1.semilogy(reg_loss_l,'--g',label='reg_loss')
            ax1.grid(True)
            ax1.set_title('Loss')
            ax1.legend(loc=3)
            # ax2.set_ylabel('Amplitude from full scale[%]', font) 

            ax2=fig.add_subplot(4,1,2)
            ax2.plot(reg_loss_l,'--g',label='reg_loss')
            ax2.plot(norm_reg_loss_l,'k',label='thick')
            ax2.plot(diff_norm_reg_loss_l,'m',label='1st diff')
            ax2.plot(laplacian_norm_reg_l, 'b',label='2nd diff')
            ax2.grid(True)
            ax2.set_title('Regularization Loss')
            ax2.legend(loc=3)


            ax3=fig.add_subplot(4,1,3)
            ax3.plot(thickness_l,'k',label='thickness')
            ax3.plot(roughness_l,'m',label='roughness')
            ax3.grid(True)
            ax3.set_title('Metric')
            ax3.legend(loc=3)
            ax3.set_ylabel('Amplitude[%]') 

            
            ax4=fig.add_subplot(4,1,4)
            ax4.plot(correct_cls_prob_l,'r',label='original cls prob')
            ax4.plot(max_prob_l,'-g',label='max prob')
            ax4.plot(prob_to_max_l,'-b',label='prob to max')
            ax4.plot(prob_to_min_l,'-k',label='prob to min')
            ax4.grid(True)
            ax4.set_title('Probability')
            ax4.legend(loc=3)
            ax4.set_ylabel('Probability') 

            
            plt.grid(True)

            # plt.draw()
            
            plt.show(block=False)
            plt.pause(0.1)


        pert = sess.run(perturbation)
        _perturbation.append(pert)

        _model_logits =  sess.run(model_logits, feed_dict={inputs: rgb_sample})




        if  step > max_step and _is_adversarial :
            res_dict['total_loss_l'] = total_loss_l
            res_dict['adv_loss_l'] = adv_loss_l
            res_dict['reg_loss_l'] = reg_loss_l
            res_dict['norm_reg_loss_l'] = norm_reg_loss_l
            res_dict['diff_norm_reg_loss_l'] = diff_norm_reg_loss_l
            res_dict['perturbation'] = _perturbation
            res_dict['adv_video'] = sess.run(adversarial_inputs_rgb, feed_dict={inputs: rgb_sample})
            res_dict['softmax'] = _model_softmax
            res_dict['total_steps'] = step
            res_dict['beta_0'] = _beta_0
            res_dict['beta_1'] = _beta_1
            res_dict['beta_2'] = _beta_2
            res_dict['beta_3'] = _beta_3
            res_dict['fatness'] = thickness_l
            res_dict['smoothness'] = roughness_l
            
            dict_result_path = os.path.join(result_path, '{}_beta1_{}_th_{:.2f}%_rg_{:.2f}%.pkl'.format(kinetics_classes[correct_cls_id].replace(' ','_'),
                                                    _beta_1,thickness_l[-1],roughness_l[-1]))

            with open(dict_result_path, 'wb') as file:
                    pickle.dump(res_dict, file)
            break

        step+=1

