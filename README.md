# Introduction

This repository contains deep learning development environment for hms project.

# Dataset cleaning

On 3080Ti mobile 16Gb:

```
for i in {0..4}; do python src/scripts/predict_val.py fit --config run/configs/common.yaml --config run/configs/timm_vit_tiny.yaml --data.init_args.by_subrecord_val true --data.init_args.batch_size 24 --data.init_args.split_index $i --data.init_args.cache_dir null; done
```

# Pseudolabling

On 3080Ti mobile 16Gb:

```
for i in {0..4}; do python run/main.py predict --config run/configs/common.yaml --config run/configs/timm_vit_tiny.yaml --config run/configs/pseudolabeling/fold_$i.yaml; done
```
