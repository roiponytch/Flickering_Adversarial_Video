# Patternless-Adversarial-Attacks-on-Video-Recognition-Networks
Code and videos accompanying the paper "Patternless Adversarial Attacks on Video Recognition Networks".


## Adversarial Video Examples
We encourage the readers to view the adversarial videos in the following:
<https://www.dropbox.com/sh/m0v5afg4zr6aypi/AAAEihgucPgGceKd3D9dBF7va?dl=0>


## Overview

Our threat model follows the white-box setting. In the experiments,
video recognition model [I3D](https://arxiv.org/abs/1705.07750) is used as target model,
focused on the RGB pipeline.
Here we attack the Inception-v1 I3D models trained on the
[Kinetics dataset](www.deepmind.com/kinetics) 

This repository contains the code to reproduce our reported results.


## Setup

This code has been tested on Ubuntu 16.04, Python 3.5.2, Tensorflow 1.15, Titan-X GPUs.

- Clone the repository 

```
git clone https://github.com/anonymousICML20/Patternless_Adversarial_Video.git
cd Patternless_Adversarial_Video

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
   
## Sample code

We provide 3 runnig scripts:

1. Single Video Attack
2. Single Class Generalization Attack
3. Universal Attack

The attack's running configuration can be modified by the file `run_config.yml`


### Single Video Attack

Configuration section `SINGLE_VIDEO_ATTACK` in `run_config.yml`
- Run `adversarial_main_single_video_npy.py`
```
Patternless_Adversarial_Video$ python adversarial_main_single_video_npy.py
```

- Visualize the adversarial video and perturbation
The result file (`.pkl`) will be save according to `PKL_RESULT_PATH` field in `run_config.yml`
for example:
```
Patternless_Adversarial_Video$ python utils/stats_and_plot/stats_plots.py result/videos_for_tests/npy/bartending_beta1_0.1_th_1.67%_rg_1.19%.pkl.pkl
```

### Single Class Generalization Attack

Download kinetics databse according to [Training Models](data/kinetics/README.md)

