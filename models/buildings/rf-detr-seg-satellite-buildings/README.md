---
library_name: transformers
license: apache-2.0
base_model: Roboflow/rf-detr-seg-medium
tags:
- instance-segmentation
- rf-detr-seg
- vision
- satellite
- building
- generated_from_trainer
model-index:
- name: rf-detr-seg-satellite-buildings
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# rf-detr-seg-satellite-buildings

This model is a fine-tuned version of [Roboflow/rf-detr-seg-medium](https://huggingface.co/Roboflow/rf-detr-seg-medium) on the merve/satellite-building-segmentation dataset.
It achieves the following results on the evaluation set:
- Loss: 36.7095

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 16
- eval_batch_size: 16
- seed: 42
- optimizer: Use OptimizerNames.ADAMW_TORCH_FUSED with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 0.1
- num_epochs: 10
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step | Validation Loss |
|:-------------:|:-----:|:----:|:---------------:|
| No log        | 1.0   | 423  | 37.8047         |
| 43.7043       | 2.0   | 846  | 37.4916         |
| 35.1590       | 3.0   | 1269 | 37.8244         |
| 33.0292       | 4.0   | 1692 | 37.4584         |
| 31.1621       | 5.0   | 2115 | 37.5637         |
| 29.3205       | 6.0   | 2538 | 37.3677         |
| 29.3205       | 7.0   | 2961 | 37.2131         |
| 27.2068       | 8.0   | 3384 | 36.6914         |
| 25.4548       | 9.0   | 3807 | 36.7667         |
| 24.4372       | 10.0  | 4230 | 36.7119         |


### Framework versions

- Transformers 5.9.0
- Pytorch 2.9.0+cu128
- Datasets 4.4.1
- Tokenizers 0.22.1
