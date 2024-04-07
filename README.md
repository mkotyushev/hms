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
for i in {0..9}; do python run/main.py         predict         --config run/configs/common.yaml         --config run/configs/timm_vit_tiny.yaml         --model.init_args.tta true     --data.init_args.by_subrecord true --data.init_args.test_is_train true --data.init_args.low_n_voters_strategy low --data.init_args.only_train true    --ckpt_path tmp/ema-0.8_$i.ckpt; mv submission.csv submission_$i.csv; done
```

# Seeds

```
python run/main.py fit --config run/configs/common.yaml --config run/configs/timm_vit_tiny.yaml --data.init_args.only_train true --seed_everything 28490463; \
python run/main.py fit --config run/configs/common.yaml --config run/configs/timm_vit_tiny.yaml --data.init_args.only_train true --seed_everything 28123213; \
python run/main.py fit --config run/configs/common.yaml --config run/configs/timm_vit_tiny.yaml --data.init_args.only_train true --seed_everything 285675463; \
python run/main.py fit --config run/configs/common.yaml --config run/configs/timm_vit_tiny.yaml --data.init_args.only_train true --seed_everything 28446341235; \
python run/main.py fit --config run/configs/common.yaml --config run/configs/timm_vit_tiny.yaml --data.init_args.only_train true --seed_everything 28491236524; \
python run/main.py fit --config run/configs/common.yaml --config run/configs/timm_vit_tiny.yaml --data.init_args.only_train true --seed_everything 1235666234; \
python run/main.py fit --config run/configs/common.yaml --config run/configs/timm_vit_tiny.yaml --data.init_args.only_train true --seed_everything 38765223; \
python run/main.py fit --config run/configs/common.yaml --config run/configs/timm_vit_tiny.yaml --data.init_args.only_train true --seed_everything 45757854; \
python run/main.py fit --config run/configs/common.yaml --config run/configs/timm_vit_tiny.yaml --data.init_args.only_train true --seed_everything 98744522; \
python run/main.py fit --config run/configs/common.yaml --config run/configs/timm_vit_tiny.yaml --data.init_args.only_train true --seed_everything 567857687; \
python run/main.py fit --config run/configs/common.yaml --config run/configs/timm_vit_tiny.yaml --data.init_args.only_train true --seed_everything 684355834
```
