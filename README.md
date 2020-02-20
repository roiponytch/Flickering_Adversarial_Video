# Patternless-Adversarial-Attacks-on-Video-Recognition-Networks
Code and videos accompanying the paper "Patternless Adversarial Attacks on Video Recognition Networks".


## Adversarial Video Examples
We encourage the readers to view the adversarial videos in the following:
<https://www.dropbox.com/sh/m0v5afg4zr6aypi/AAAEihgucPgGceKd3D9dBF7va?dl=0>


## Overview

Our threat model follows the white-box setting. In the experiments,
video recognition model I3D "(https://arxiv.org/abs/1705.07750)" is used as target model,
focused on the RGB pipeline.
Here we attack the Inception-v1 I3D models trained on the
[Kinetics dataset](www.deepmind.com/kinetics) 

This repository contains the code to reproduce our reported results.


## Running the code

### Setup

1. Clone this repository using
 
   `$ git clone https://github.com/anonymousICML20/Patternless_Adversarial_Video.git`

2. To install the dependencies, run

   `$ pip install -r requirements.txt`

And for the GPU to work, make sure you've got the drivers installed beforehand (CUDA).

It has been tested to work with Python 3.5.2.

3. Download and merge the following diractory to project dir https://www.dropbox.com/sh/ilbsy3bwk5k5tn4/AADxk11U_EDalu467igLfX2wa?dl=0.
