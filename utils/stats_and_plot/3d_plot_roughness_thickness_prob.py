#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 18:59:22 2020

@author: ubadmin
"""
# %%

from mpl_toolkits.mplot3d import Axes3D
ax = fig.add_subplot(111, projection='3d')
z = np.max(tmp_struct.softmax, axis=-1)

fig = plt.figure()
ax = fig.gca(projection='3d')
results_path = '/data/DL/Adversarial/kinetics-i3d/result/videos_for_tests/npy/rgb_sIn7Te48YL4@shooting_goal_(soccer)/'

leg=[]
allfiles = os.listdir(results_path)
files = [fname for fname in allfiles if fname.endswith('.pkl')]
beta_1 = [float(f.split('_')[-1].strip('.pkl')) for f in files]
idx = np.argsort(np.array(beta_1)).astype(np.uint8)
col = np.linspace(0.1,1,files.__len__())

for i,res in enumerate(files):
    fid = 'rgb_sIn7Te48YL4@shooting_goal_(soccer)beta_1_0.' + str(i) + '0.pkl'
    if i ==10: fid = 'rgb_sIn7Te48YL4@shooting_goal_(soccer)beta_1_1.00.pkl'
    
    with open(os.path.join(results_path,fid), 'rb') as handle:
        tmp_dict = pickle.load(handle)

    tmp_dict['softmax'] = np.concatenate(tmp_dict['softmax'], axis=0)
    tmp_struct = namedtuple("dict", tmp_dict.keys())(*tmp_dict.values())

    z = np.max(tmp_struct.softmax, axis=-1)
    smoothness =np.mean(np.mean(np.abs(np.array(tmp_struct.perturbation)-np.roll(np.array(tmp_struct.perturbation),1,axis=1)),axis=1),-1)
    fatness =np.mean(np.mean(np.abs(np.array(tmp_struct.perturbation)),axis=1),-1)
    
    smoothness_ = smoothness.squeeze()/ 2.0 * 100
    fatness_ = fatness.squeeze()/ 2.0 * 100

    ax.plot(smoothness_, fatness_, z,color=[col[i],1-col[i],0])
    leg.append(r'$\beta_1$: {:.1f} $\beta_2$: {:.1f}' .format(i/10,1-i/10))
   
ax.legend(leg)
plt.xlabel('Roughness')
plt.ylabel('Thickness')
ax.set_zlabel('probability')



