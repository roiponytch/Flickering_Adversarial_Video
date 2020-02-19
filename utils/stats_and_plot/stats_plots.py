#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 18:59:22 2020

@author: ubadmin
"""
# %%
import os
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pickle
from collections import namedtuple
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import utils.kinetics_i3d_utils as ki3du

#%%
    
def main(argv, arc):
    
    kinetics_classes =ki3du.load_kinetics_classes()
    path = argv[1]

    save_to_vid = 0
    # path ='/data/DL/Adversarial/kinetics-i3d/result/videos_for_tests/npy/kinetics@triple_jumpbeta_1_0.50.pkl'
    # path='/data/DL/Adversarial/kinetics-i3d/result/videos_for_tests/pkl_final/rgb_sIn7Te48YL4@shooting_goal_(soccer)/rgb_sIn7Te48YL4@shooting_goal_(soccer)beta_1_1.00.pkl'
    with open(path, 'rb') as handle:
        tmp_dict = pickle.load(handle)
    
    tmp_dict['softmax'] = np.concatenate(tmp_dict['softmax'], axis=0)
    tmp_struct = namedtuple("dict", tmp_dict.keys())(*tmp_dict.values())
    
    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure(facecolor='black')#,constrained_layout=True)
    
    ax_pert_graph =fig.add_subplot(2,3,5,facecolor='k')
    ax_pert_graph.set_xlim((0, 90))
    
    
    ax_adv_vid=fig.add_subplot(2,3,3)
    ax_pert_vid=fig.add_subplot(2,3,2)
    ax_cln_vid=fig.add_subplot(2,3,1)
    # ax1 = ax.twinx()
    
    # ax = plt.axes(xlim=(0, 90), ylim=(-0.1, 0.1))
    ax_adv_vid.axis('OFF')
    ax_pert_vid.axis('OFF')
    ax_cln_vid.axis('OFF')
    
    
    line, = ax_pert_graph.plot([],[] ,lw=2)
    
    adv_video = ((tmp_struct.adv_video[0] +1.0)*127.5).astype(np.uint8)
    dummy_img = adv_video[0]
    cln_video = ((tmp_struct.rgb_sample[0] +1.0)*127.5).astype(np.uint8)
    
    pert_raw=tmp_struct.perturbation[-1].copy()-tmp_struct.perturbation[-1].min()
    
    scale_factor = int(2/pert_raw.max())
    pert_raw/=pert_raw.max()
    pert_raw*=255
    pert_raw=pert_raw.astype(np.uint8)
    
    
    pert_video=np.repeat(pert_raw,224,axis=1)
    pert_video=np.repeat(pert_video,224,axis=2)
    
    
    pert = tmp_struct.perturbation[-1].squeeze()/2.0*100
    
    font = {'family': 'serif',
            'color':  'white',
            'weight': 'normal',
            'size': 16,
            }
    
    # mng = plt.get_current_fig_manager()
    # mng.full_screen_toggle()
    # manager = plt.get_current_fig_manager()
    # manager.window.showMaximized()
    
    ax_cln_vid.set_title('Clean video\n top-1 class: {}'.format(kinetics_classes[tmp_struct.correct_cls_id]) ,font)
    ax_pert_vid.set_title('Perturbation\n'+r'(amplified $\times${} for visualization)'.format(scale_factor),font)
    ax_adv_vid.set_title('Adversarial video\n top-1 class: {}'.format(kinetics_classes[tmp_struct.softmax[-1].argmax()]),
                              font)
    
    ax_pert_graph.set_title('RGB Perturbation\n percents from the full scale of the image',font)
    ax_pert_graph.set_ylabel('Amplitude from full scale[%]', font) 
    
    font2 = {'family': 'serif',
            'color':  'y',
            'weight': 'normal',
            'size': 16,
            }
    ax_pert_graph.set_xlabel('Current\nperturbation', font2) 
    
    
    y_top=1.2*np.abs(pert).max()
    ax_pert_graph.set_ylim(-y_top,y_top)
    
    # ax_pert_graph.yaxis.label.set_color('white')
    ax_pert_graph.tick_params(axis='y', labelcolor='w')
    ax_pert_graph.tick_params(axis='x', colors='k')
    ax_pert_graph.grid(True)
    
    
    pp=y_top-np.abs(pert).max()
    
    ax_pert_graph.arrow(45,-y_top, 0, 0.5*pp, head_width=2, head_length=0.5*pp, fc='y', ec='y')
    
    ax_pert_graph.arrow(45,y_top, 0, -0.5*pp, head_width=2, head_length=0.5*pp, fc='y', ec='y')
    
    
    # plt.tight_layout()
    # ax_pert_graph.annotate('a polar annotation',
    #             xy=(45, -1),# theta, radius
    #             xytext=(0.5, 1),    # fraction, fraction
    #             textcoords='figure fraction',
    #             arrowprops=dict(facecolor='white', shrink=0.05),
    #             horizontalalignment='left',
    #             verticalalignment='bottom')
    # ax_pert_graph.spines['left'].set_color('w')
    fig.set_size_inches(19, 11)
    # ax_pert_graph.spines['left'].set_color('white')
    # plt.rc('axes',edgecolor='white')
    
    
    img_adv=ax_adv_vid.imshow(np.zeros_like(dummy_img,dtype=np.uint8),zorder=1)
    img_cln=ax_cln_vid.imshow(np.zeros_like(dummy_img,dtype=np.uint8),zorder=1)
    img_pert=ax_pert_vid.imshow(np.zeros_like(dummy_img,dtype=np.uint8),zorder=1)
    
    plus_pos = [(ax_cln_vid.get_position().x1 + ax_pert_vid.get_position().x0)/2,
                (ax_cln_vid.get_position().y1+ax_cln_vid.get_position().y0)/2]
    
    fig.text(plus_pos[0],plus_pos[1],'$+$',horizontalalignment='center',verticalalignment='center',fontsize=18,color='white')
    
    equal_pos = [(ax_pert_vid.get_position().x1 + ax_adv_vid.get_position().x0)/2,
                (ax_pert_vid.get_position().y1+ax_pert_vid.get_position().y0)/2]
    
    fig.text(equal_pos[0],equal_pos[1],'$=$', horizontalalignment='center',verticalalignment='center',fontsize=18,color='white')
    
    lines = []
    plotlays, plotcols = [3], ["red","green","blue"]
    
    roughness=tmp_struct.smoothness[-1]
    thickness=tmp_struct.fatness[-1]
    
    beta1=tmp_struct.beta_1
    
    if hasattr(tmp_struct, 'beta_3'):
        beta2 = tmp_struct.beta_2 +tmp_struct.beta_3
    else:
         beta2 = tmp_struct.beta_2*2
    
    fig.suptitle('Adversarial example: '+r'$\beta_1$={},$\beta_2$={},'.format(beta1,beta2)
                  +' Thickness={:.2f}%, Roughness={:.2f}%'.format(thickness,roughness),color='w',fontsize=16)
    fig.subplots_adjust(hspace=0.22)
    
    # plt.text(10,550 , 'I. Naeh, R. Pony, S. Mannor \"Patternless Adversarial Attacks on Video Recognition Networks\" arXiv',
    #         verticalalignment='bottom', horizontalalignment='right',
    #         color='green', fontsize=15)
    
    for index in range(3):
        lobj = ax_pert_graph.plot([],[],lw=2,color=plotcols[index])[0]
        lines.append(lobj)
    
    # initialization function: plot the background of each frame
    def init():
        
        for i in lines:
            line.set_data([],[])
            
        img_adv.set_data(np.zeros_like(dummy_img,dtype=np.uint8))
        img_cln.set_data(np.zeros_like(dummy_img,dtype=np.uint8))
        img_pert.set_data(np.zeros_like(dummy_img,dtype=np.uint8))
    
        
        return lines
    
    # animation function.  This is called sequentially
    def animate(i):
        ii=i %90
        
        x = np.linspace(0, 89, 90)
        y=np.roll(pert,-ii-45,0)
    
        img_adv.set_data(adv_video[ii])
        img_cln.set_data(cln_video[ii])
        img_pert.set_data(pert_video[ii])
    
        y_mean=y.mean(axis=-1)
        y_std = y.std(axis=-1)
        
        
        for lnum,line in enumerate(lines):
            line.set_data(x,y[...,lnum]) # set data for each line separately. 
    
        # p1 = ax_pert_graph.fill_between(np.arange(50,91) , -y_top,y_top, facecolor = 'gray', alpha = 0.2)
        # p2 = ax_pert_graph.fill_between(np.arange(0,41) , -y_top,y_top, facecolor = 'gray', alpha = 0.2)
        # p3 = ax_pert_graph.fill_between(np.arange(45,47) , -y_top,y_top, facecolor = 'y', alpha = 0.5)
    
        return lines[0],lines[1],lines[2],img_adv,img_cln,img_pert#,p1,p2#img, #lines
    
    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init, save_count=900,
                                   frames=90*3, interval=100, blit=True,repeat=True)
    
    # save the animation as an mp4.  This requires ffmpeg or mencoder to be
    # installed.  The extra_args ensure that the x264 codec is used, so that
    # the video can be embedded in html5.  You ma need to adjust this for
    # your system: for more information, see
    # http://ma{{tplotlib.sourceforge.net/api/animation_api.html
    if save_to_vid:
        anim.save('{}_beta1_{}_th_{:.2f}%_rg_{:.2f}%.mp4'.format(kinetics_classes[tmp_struct.correct_cls_id].replace(' ','_'),
                                                        tmp_struct.beta_1,thickness,roughness), fps=12,dpi=100, extra_args=['-vcodec', 'libx264','-crf', '5'],savefig_kwargs={'bbox_inches':'tight','quality':100,'facecolor':'black'}) #'-filter_complex','loop=loop=3:size=270:start=0'])
    plt.show()
    
if __name__ == '__main__':
    main(sys.argv, len(sys.argv))