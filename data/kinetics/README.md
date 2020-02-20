# Kinetics - Downloader

## Usage

Setup your environment
```
pip install --upgrade youtube-dl
```

Download a dataset split by calling:
```
mkdir <data_dir>
python download.py {dataset_split}.csv <data_dir> -n <num-jobs>
```
for example:
```
mkdir database/test/
python download.py 'data/kinetics-400_test.csv' 'database/test/' -n 32
```
