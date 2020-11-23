# Over-the-Air Adversarial Flickering Attacks against Video Recognition Networks
Code and videos accompanying the paper "Over-the-Air Adversarial Flickering Attacks against Video Recognition Networks".


## Adversarial Video Examples
We encourage the readers to view the adversarial videos in the following:
https://bit.ly/Flickering_Attack_videos
https://bit.ly/Over_the_Air_videos
https://bit.ly/Over_the_Air_scene_based_videos


## Overview
<!---
![](bartending_beta1_0.1_th_1.67__rg_1.19.gif)
-->

Our threat models follows the white-box setting. In the experiments we used the [I3D](https://arxiv.org/abs/1705.07750) , [R(2+1)D, R3D and MC3](https://arxiv.org/abs/1711.11248) as target models (pre trained on the [Kinetics dataset](www.deepmind.com/kinetics) ).

This repository contains the code to reproduce our reported results.


## Setup

This code has been tested on Ubuntu 16.04, Python 3.7.5, Tensorflow 1.15, torch  1.4.0 and Titan-X GPUs.

- Clone the repository 

```
git clone https://github.com/anonymous-p/Flickering_Adversarial_Video
cd Flickering_Adversarial_Video

```

- Install the dependencies
```
pip install -r requirements.txt
```

- Download and merge checkpoint and additional [data](https://www.dropbox.com/sh/ilbsy3bwk5k5tn4/AADxk11U_EDalu467igLfX2wa?dl=0) 
   
```
./download_ckpt_and_data.sh
mv data/result/ .
```
   
## Sample code I3D

We provide 3 runnig scripts:

1. Single Video Attack
2. Single Class Generalization Attack
3. Universal Attack

The attack's running configuration can be modified by the file `run_config.yml`


### Single Video Attack

Configuration section `SINGLE_VIDEO_ATTACK` in `run_config.yml`
- Run `adversarial_main_single_video_npy.py`
```
Flickering_Adversarial_Video$ python i3d_adversarial_main_single_video_npy.py
```

- Visualize the adversarial video and perturbation

The result file (`.pkl`) will be save according to `PKL_RESULT_PATH` field in `run_config.yml`.

We provide example of `.pkl`:
```
Flickering_Adversarial_Video$ python utils/stats_and_plot/stats_plots.py result/videos_for_tests/npy/bartending_beta1_0.1_th_1.67%_rg_1.19%.pkl
```

### Single Class Generalization Attack

- Download kinetics databse according to [Kinetics-Downloader](data/kinetics/README.md)
- Convert the downloaded data to `.tfrecord`:

```
 python kinetics_to_tf_record_uint8.py <data_dir> class <tfrecord_data_dir> 
```
`<data_dir>` : the source (.mp4) data dir 

`class` : class to convert

 `<tfrecord_data_dir>` : destination path to save `.tfrecord`'s
 

for example, convert only `triple jump` in val split to tfrecord:  

```
 python kinetics_to_tf_record_uint8.py 'data/kinetics/database/val/' 'triple jump' 'data/kinetics/database/tfrecord_uint8/val/' 
```

for example, convert all class in val split to tfrecord:  

```
 python kinetics_to_tf_record_uint8.py 'data/kinetics/database/val/' 'all' 'data/kinetics/database/tfrecord_uint8/val/' 
```

Configuration section `CLASS_GEN_ATTACK` in `run_config.yml`
- Run `adversarial_main_single_class_gen.py`
```
Flickering_Adversarial_Video$ python i3d_adversarial_main_single_class_gen.py
```

### Universal Attack

- Download kinetics databse according to [Kinetics-Downloader](data/kinetics/README.md)
- Convert the downloaded data to `.tfrecord` s.t create tfrecord's with shuffled classes':

```
 python kinetics_to_tf_record_uint8_shuffle.py 'data/kinetics/database/test/' 'data/kinetics/database/tfrecord/test_all_cls/' 
```
Configuration section `UNIVERSAL_ATTACK` in `run_config.yml`
- Run `adversarial_main_universal.py`
```
Flickering_Adversarial_Video$ python i3d_adversarial_main_universal.py
```
In order to run [Sparse Adversarial Perturbations for Videos]( https://arxiv.org/pdf/1803.02536.pdf) 
set `FLICKERING_ATTACK` to `False`

Loss's and metric's can be monitor with 'tensorboard', for example:
```
Flickering_Adversarial_Video$ tensorboard --logdir=result/generalization/universal_untargeted/
```
## Sample code R(2+1)D, R3D and MC3 [Sparse Adversarial Perturbations for Videos]( https://arxiv.org/pdf/1803.02536.pdf) 
We provide 2 runnig scripts:

1. Single Video Attack
2. Universal Attack

Before using the running scripts the kinetics databse shuold be downloaded according to [Kinetics-Downloader](data/kinetics/README.md).

The attack's running configuration can be modified inside the running script under 'MODEL AND TRAIN PARAMETERS', allowing to set the attacked model ( R(2+1)D, R3D and MC3) and other parameters regarding the attack.

### Single Video Attack

- Run `r2plus1d_main_statistics_single_video_attack.py`
```
Flickering_Adversarial_Video$ python r2plus1d_main_statistics_single_video_attack.py
```
### Universal Attack

- Run `r2plus1d_main_universal_attack.py`
```
Flickering_Adversarial_Video$ python r2plus1d_main_universal_attack.py
```
