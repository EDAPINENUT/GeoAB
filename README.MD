# Installation
Install requirements

```
conda create -n geoab python==3.9
conda activate geoab
pip install -r requirements.txt
```
# Dataset
Please follow the data preparation scripts in DyMean, which leads the raw data set as 
```
- all_structures
    - chothia
    - imgt
    - raw
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
The processed data will be uploaded soon. Since the double-bind review, and it is hard to find an anonymouls
## Train and evaluate GeoAB-Refiner
Run the following command for training:

```
# Train GeoAB-refiner
python train_refine.py

# Train GeoAB-Initializer
python train_init.py
# After GeoAB-Initializer is trained, train GeoAB-Designer
python train_design.py
```

For evaluation, run the following command for evaluation:

```
# Evaluate GeoAB-Refiner
python eval.py --eval_dir H3_refine --run 1

# Evaluate GeoAB-Designer
python eval.py --eval_dir H3_design
```

We give the pretrained model in `./results/cdrh3/H3_refine/checkpoint/best.ckpt`, `/linhaitao/GeoAB/results/cdrh3/H3_init/checkpoint/best.ckpt` and `./results/cdrh3/H3_design/checkpoint/best.ckpt`. You can evaluate the pretrained ones directly using our pretrained models.