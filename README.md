# Introduction

This repository contains deep learning development environment for hms project.

# Dataset cleaning

On 3080Ti mobile 16Gb:

```
for i in {0..4}; do python src/scripts/predict_val.py fit --config run/configs/common.yaml --config run/configs/timm_vit_tiny.yaml --data.init_args.by_subrecord_val true --data.init_args.batch_size 24 --data.init_args.split_index $i --data.init_args.cache_dir null; done
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

*Note: seed 28446341235 and seed 28491236524 somehow yield same results, so 28446341235's predictions are not used as PLs (`submission_2.csv` is not included in `pl_filepathes`) and the seed is not used for training with PLs.*

Resulting models (`ema-0.8_$i.ckpt`) are moved to `tmp/` dir.

# Pseudolabels generation

```
for i in {0..10}; do python run/main.py         predict         --config run/configs/common.yaml         --config run/configs/timm_vit_tiny.yaml         --model.init_args.tta true     --data.init_args.by_subrecord true --data.init_args.test_is_train true --data.init_args.low_n_voters_strategy low --data.init_args.only_train true    --ckpt_path tmp/ema-0.8_$i.ckpt; mv submission.csv submission_$i.csv; done
```

Resulting PLs (`submission_$i.csv`) files are moved to `labels/pseudolabel/` dir.

# Seeds with PL

```
for s in 28490463 28123213 285675463 28491236524 1235666234 38765223 45757854 98744522 567857687 684355834; do python run/main.py fit --config run/configs/common.yaml --config run/configs/timm_vit_tiny.yaml --data.init_args.only_train true --data.init_args.low_n_voters_strategy both --data.init_args.pl_other_vote_threshold 0.8 --data.init_args.pl_filepathes '["labels/pseudolabel/submission_0.csv", "labels/pseudolabel/submission_1.csv", "labels/pseudolabel/submission_3.csv", "labels/pseudolabel/submission_4.csv", "labels/pseudolabel/submission_5.csv", "labels/pseudolabel/submission_6.csv", "labels/pseudolabel/submission_7.csv", "labels/pseudolabel/submission_8.csv", "labels/pseudolabel/submission_9.csv", "labels/pseudolabel/submission_10.csv"]' --seed_everything $s; done
```
