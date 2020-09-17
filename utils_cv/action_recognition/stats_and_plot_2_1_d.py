
##
import numpy as np
import glob
import os
import re
import matplotlib.pyplot as plt

DL_shared= '{}/DL_shared'.format(os.path.expanduser('~'))
computervision_recipes_path =DL_shared+'/Adversarial/computervision-recipes/'

##
results_path=computervision_recipes_path + '/results/r2plus1d_18/single_video_attack/train/all_cls_shuffle_flickering/lambda_1.0_beta1_0.5_/'

all_npy_path = glob.glob(results_path+'*.npy')



thickness_l=[]
roughness_l=[]
fooling_l = []

for path in all_npy_path:
    res_dict = np.load(path,allow_pickle=True)
    
    res_dict = res_dict.tolist()
    if res_dict is None:
        continue
    roughness = np.array(res_dict['perturbation/roughness'])
    thickness = np.array(res_dict['perturbation/thickness'])
    
    is_adversarial = np.array(res_dict['is_adversarial'])

    fooling = is_adversarial.any()
    fooling_l.append(fooling)
    
    if fooling:
        min_idx = thickness[is_adversarial].argmin()
        
        thickness_l.append(thickness[is_adversarial][min_idx])
        roughness_l.append(roughness[is_adversarial][min_idx])

fooling_ratio = np.sum(fooling_l)/len(fooling_l)
## universal attack - by num of videos for train

results_paths=[]
lns_metric=[]
lns_fool=[]
colors=['b','r','g','c']

thick_dict={}
rough_dict={}
fooling_dict={}
# results_paths.append(computervision_recipes_path + '/results/r2plus1d_34_32_kinetics/generalization/universal/val_test/all_cls_shuffle_flickering/')
results_paths.append(computervision_recipes_path + '/results/r2plus1d_18/generalization/universal/val_test/all_cls_shuffle_flickering/')
results_paths.append(computervision_recipes_path + '/results/mc3_18/generalization/universal/val_test/all_cls_shuffle_flickering/')
results_paths.append(computervision_recipes_path + '/results/r3d_18/generalization/universal/val_test/all_cls_shuffle_flickering/')

fig, ax1 = plt.subplots(1, 1)
ax1.set_xlabel('Num_of_videos (#)')
ax1.set_ylabel('Fooling_ratio', color='g')
fig2,ax2 = plt.subplots(1, 1)

ax2.set_ylabel('[%]', color='b')

ax2.tick_params(axis='y', labelcolor='b')
ax1.tick_params(axis='y', labelcolor='g')
for color,results_path in zip(colors,results_paths):
    model_name = re.search('results/(.+?)/',results_path).group(1)

    all_exp_path = glob.glob(results_path + '*')

    thickness_l = []
    roughness_l = []
    fooling_l = []
    num_train_vid_l = []

    for path in all_exp_path:

        exp_folder = path.split('/')[-1]

        num_train_videos = int(re.search('t_(.+?)_',exp_folder).group(1))
        num_train_vid_l.append(num_train_videos)
        num_val_videos = int(re.search('v_(.+?)_', exp_folder).group(1))
        l_inf = float(re.search('linf_(.+?)_', exp_folder).group(1))
        lambda_param = float(re.search('lambda_(.+?)_', exp_folder).group(1))
        beta1_param = float(re.search('beta1_(.+?)_', exp_folder).group(1))

        ckpt_paths = glob.glob(path + '/*.npy')
        ckpt_paths.sort(key=os.path.getmtime)

        res_dict = np.load(ckpt_paths[-1], allow_pickle=True)[-1]
        fooling_l.append(res_dict['valid/fooling_ratio'].data.cpu().numpy().item())
        thickness_l.append(res_dict['valid/pert_thickness'])
        roughness_l.append(res_dict['valid/pert_roughness'])

    thick_dict[model_name]=thickness_l
    rough_dict[model_name]=roughness_l
    fooling_dict[model_name]=fooling_l

    sort_idx_by_num_of_vid=np.argsort(num_train_vid_l)

    num_train_vid_l = np.array(num_train_vid_l)[sort_idx_by_num_of_vid]
    fooling_l = np.array(fooling_l)[sort_idx_by_num_of_vid]
    thickness_l = np.array(thickness_l)[sort_idx_by_num_of_vid]
    roughness_l = np.array(roughness_l)[sort_idx_by_num_of_vid]


     # instantiate a second axes that shares the same x-axis

    lns1 = ax1.semilogx(num_train_vid_l,fooling_l,label='Fooling_ratio_{}'.format(model_name),color=color)

    lns2= ax2.semilogx(num_train_vid_l,thickness_l*100, color, label='thickness_{}'.format(model_name))
    lns3= ax2.semilogx(num_train_vid_l,roughness_l*100, '--'+color, label='roughness_{}'.format(model_name))

    lns_fool.append(lns1)
    lns_metric.append(lns2)
    lns_metric.append(lns3)


lns = lns_fool[0]+lns_fool[1]+lns_fool[2]
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=6)
plt.title('Untarget Universal attack')
ax1.grid(True)

lns = lns_metric[0] + lns_metric[1] +lns_metric[2] +lns_metric[3] + lns_metric[4] +lns_metric[5]
labs = [l.get_label() for l in lns]
ax2.legend(lns, labs, loc=6)
plt.title('Untarget Universal attack')
ax2.grid(True)


##

