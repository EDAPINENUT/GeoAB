# GeoAB:  Towards Realistic Antibody Design and Reliable Affinity Maturation
## Installation
Install requirements

```
conda create -n geoab python==3.9
conda activate geoab
pip install -r requirements.txt
```
## Dataset
Please follow the data preparation scripts in DyMean, which leads the raw data set as 
```
- all_data
    - RAbD_H3
        - test_processed
            _metainfo
            part_0.pkl
        - train_processed
            ...
        - valid_processed
            ...
        test.json
        train.json
        valid.json
    - SKEMPI
        ...
```
The processed data can be downloaded from google drive https://drive.google.com/drive/folders/1pNsoGt0gDIJR9EMmmp2pItjZDuLZI3gg. 
After downloading `all_data.zip`, unzip it and the processed datasets can be obtained.

## Train and evaluate GeoAB-Designer
Run the following command for training:

```
# Train GeoAB-refiner
python train_refine.py

# Train GeoAB-Initializer
python train_init.py
# After GeoAB-Initializer is trained, train GeoAB-Designer
python train_design.py
```

For evaluation, run the following command:

```
# Evaluate GeoAB-Refiner
python eval.py --eval_dir H3_refine --run 1

# Evaluate GeoAB-Designer
python eval.py --eval_dir H3_design
```

We give the pretrained model in `cdrh3.zip`, which can be downloaded from https://drive.google.com/drive/folders/1pNsoGt0gDIJR9EMmmp2pItjZDuLZI3gg. You can evaluate the results directly using our pretrained models.


## DDG Prediction
For DDG prediction part, our model will be updated through a platform, which will be online soon.

# Citation
Please cite the paper if the repository or the paper is helpful to you, as the following
```
@article {lin2024geoab,
	author = {Lin, Haitao and Wu, Lirong and Huang, Yufei and Liu, Yunfan and Zhang, Odin and Zhou, Yuanqing and Sun, Rui and Li, Stan Z.},
	title = {GeoAB: Towards Realistic Antibody Design and Reliable Affinity Maturation},
    	year = {2024},
	booktitle={International Conference on Machine Learning},
	URL = {https://www.biorxiv.org/content/early/2024/05/17/2024.05.15.594274},
	eprint = {https://www.biorxiv.org/content/early/2024/05/17/2024.05.15.594274.full.pdf}
}

```
