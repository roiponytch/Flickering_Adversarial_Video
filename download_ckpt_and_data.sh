#! /bin/bash
wget https://www.dropbox.com/sh/tllkruey72phm6q/AACqBb2fdeCvUHcYgTIsNzM3a?dl=0 -O ckpt_and_data.tar.gz
unzip ckpt_and_data.tar.gz -d data/
rm ckpt_and_data.tar.gz
mv data/result .