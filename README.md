# Playbook: Scalable Discrete Skill Discovery from Unstructured Datasets for Long-Horizon Decision-Making Problems

This repository is the official implementation of "Playbook: Scalable Discrete Skill Discovery from Unstructured Datasets for Long-Horizon Decision-Making Problems". 

## Implementation Reference

1. Implementation of Trajectory Transformer (./models/high_level_dynamics)
> In this repository, we modify and use the official implementation of the authors of "Trajectory Transformer" (link: https://github.com/jannerm/trajectory-transformer).

2. CALVIN environment (./calvin)
> In this repository, we modify and use the official implementation of the "CALVIN" environment (link: https://github.com/mees/calvin).

3. TACO-RL dataset
> In this repository, we use the offline dataset for CALVIN provided by the "TACO-RL" authors (link: https://github.com/ErickRosete/tacorl).
> To train or evaluate a "Playbook" model in CALVIN, please download the "Environment D" dataset in the above webpage.
> Use the downloaded path as an input of the "data_dir" argument.


## Requirements

We use python3.8 for the playbook implementation.

1. Requirements for the playbook code
```
cd Playbook
pip install -r requirements.txt
```

2. Requirements for Trajectory Transformer
```
cd Playbook/models/high_level_dynamics
pip install -e .
```

3. Requirements for CALVIN
> Download the calvin folder file [https://drive.google.com/file/d/1GehhoL8Ntfc4oC3-nQSxT-x7vv7HN0nn/view?usp=sharing] and locate it in the main playbook folder.
```
cd Playbook/calvin/calvin_env/tacto
pip install -e .
cd ../
pip install -e .
cd ../calvin_model
pip install -e . --find-links https://download.pytorch.org/whl/torch_stable.html
```
> You must have ".git" folder in "./calvin". If it is lost, please generate it through the "git clone --recurse-submodules https://github.com/mees/calvin.git" command.

## Training

1. Franka Kitchen Environment

1.1. Low-Level Models
```
python train_low_level.py --task kitchen-partial --filename test1 --n_subpols [16] --n_weights [32] --z_dep_dim 32 --z_ind_dim 16 --total_steps 200000

```
1.2. Trajectory Generation Model (Trajectory Transformer)
```
python train_high_level.py --task kitchen-partial --filename test1 --loadname test1_play32_subpol16_LS32_LA16_H10 --n_subpols [16] --n_weights [32] --z_dep_dim 32 --z_ind_dim 16 --work dynamics
```
1.3. State Distance Metric Model (Implicit Q-Learning)
```
python train_high_level.py --task kitchen-partial --filename test1 --loadname test1_play32_subpol16_LS32_LA16_H10 --n_subpols [16] --n_weights [32] --z_dep_dim 32 --z_ind_dim 16 --work distance
```

2. CALVIN Environment

2.1. Low-Level Models
```
python train_low_level.py --task calvin --filename test1 --n_subpols [32] --n_weights [64] --z_dep_dim 64 --z_ind_dim 32 --total_steps 300000
```
2.2. Trajectory Generation Model (Trajectory Transformer)
```
python train_high_level.py --task calvin --filename test1 --loadname test1_play64_subpol32_LS64_LA32_H10 --n_subpols [32] --n_weights [64] --z_dep_dim 64 --z_ind_dim 32 --work dynamics
```
2.3. State Distance Metric Model (Implicit Q-Learning)
```
python train_high_level.py --task calvin --filename test1 --loadname test1_play64_subpol32_LS64_LA32_H10 --n_subpols [32] --n_weights [64] --z_dep_dim 64 --z_ind_dim 32 --work distance
```

## Training for Playbook Extension

1. Low-Level Models

1.1. Step 0 (using base dataset)
```
python train_low_level.py --task calvin --filename cont1 --n_subpols [32] --n_weights [64] --z_dep_dim 64 --z_ind_dim 32 --do_continual 1 --use_newdata 0 --do_distill 0 --continual_step 1 --total_steps 300000
```
1.2. Step 1: Extension (using close-drawer dataset)
```
python train_low_level.py --task calvin --filename cont1 --loadname cont1_play64_subpol32_LS64_LA32_H10 --n_subpols [32,2] --n_weights [64,4] --z_dep_dim 64 --z_ind_dim 32 --do_continual 1 --use_newdata 1 --do_distill 0 --continual_step 1 --remaining_ratio 0.005 --total_steps 200000
```
1.3. Step 1: Distillation (using close-drawer dataset)
```
python train_low_level.py --task calvin --filename cont1 --loadname cont1_play64_subpol32_LS64_LA32_H10 --n_subpols [32,2] --n_weights [64,4] --z_dep_dim 64 --z_ind_dim 32 --do_continual 1 --use_newdata 1 --do_distill 1 --continual_step 1 --remaining_ratio 0.005 --total_steps 100000
```
1.4. Step 2: Extension (using move-slider-left dataset)
```
python train_low_level.py --task calvin --filename cont1 --loadname cont1_play64_subpol32_LS64_LA32_H10 --n_subpols [32,2,2] --n_weights [64,4,4] --z_dep_dim 64 --z_ind_dim 32 --do_continual 1 --use_newdata 1 --do_distill 0 --continual_step 2 --remaining_ratio 0.005 --total_steps 200000
```
1.5. Step 2: Distillation (using move-slider-left dataset)
```
python train_low_level.py --task calvin --filename cont1 --loadname cont1_play64_subpol32_LS64_LA32_H10 --n_subpols [32,2,2] --n_weights [64,4,4] --z_dep_dim 64 --z_ind_dim 32 --do_continual 1 --use_newdata 1 --do_distill 1 --continual_step 2 --remaining_ratio 0.005 --total_steps 100000
```
> For subsequent steps, you can change "n_subpols", "n_weights", and "continual_step" arguments according to the current continual learning step.


2. Trajectory Transformer Model

2.1. Step 0 (using base dataset)
```
python train_high_level.py --task calvin --filename cont1 --loadname cont1_play64_subpol32_LS64_LA32_H10 --n_subpols [32] --n_weights [64] --z_dep_dim 64 --z_ind_dim 32 --do_continual 1 --use_newdata 0 --continual_step 1 --work dynamics
```
2.2. Step 1 (using close-drawer dataset)
```
python train_high_level.py --task calvin --filename cont1 --loadname cont1_play64_subpol32_LS64_LA32_H10 --n_subpols [32,2] --n_weights [64,4] --z_dep_dim 64 --z_ind_dim 32 --do_continual 1 --use_newdata 1 --continual_step 1 --work dynamics
```
> For subsequent steps, you can change "n_subpols", "n_weights", and "continual_step" arguments according to the current continual learning step.


3. State Distance Metric Model

3.1. Step 0 (using base dataset)
```
python train_high_level.py --task calvin --filename cont1 --loadname cont1_play64_subpol32_LS64_LA32_H10 --n_subpols [32] --n_weights [64] --z_dep_dim 64 --z_ind_dim 32 --do_continual 1 --use_newdata 0 --continual_step 1 --work distance
```
3.2. Step 1 (using close-drawer dataset)
```
python train_high_level.py --task calvin --filename cont1 --loadname cont1_play64_subpol32_LS64_LA32_H10 --n_subpols [32,2] --n_weights [64,4] --z_dep_dim 64 --z_ind_dim 32 --do_continual 1 --use_newdata 1 --continual_step 1 --work distance
```
> For subsequent steps, you can change "n_subpols", "n_weights", and "continual_step" arguments according to the current continual learning step.


## Evaluation

1. Franka Kitchen Experiment
```
python test.py --task kitchen-partial  --loadname test1_play32_subpol16_LS32_LA16_H10 --n_subpols [16] --n_weights [32] --z_dep_dim 32 --z_ind_dim 16 --eval_episodes 100
python test.py --task kitchen-mixed  --loadname test1_play32_subpol16_LS32_LA16_H10 --n_subpols [16] --n_weights [32] --z_dep_dim 32 --z_ind_dim 16 --eval_episodes 100
```
> You can change "task" argument: kitchen-partial or kitchen-mixed.

2. CALVIN Experiment (No Extension)
```
python test.py --task calvin  --loadname test1_play64_subpol32_LS64_LA32_H10 --n_subpols [32] --n_weights [64] --z_dep_dim 64 --z_ind_dim 32 --eval_type in_a_row --len_task_chain 3 --data_dir [tacorl_dataset_dir]
```
> You can change "eval_type" (in_a_row or individually) and "len_task_chain" (1, 2 or 3) arguments.

3. CALVIN Extension Experiment
```
python continual_test.py --task calvin --loadname cont1_play64_subpol32_LS64_LA32_H10 --n_subpols [32,2,2,2,2] --n_weights [64,4,4,4,4] --z_dep_dim 64 --z_ind_dim 32 --eval_type in_a_row --len_task_chain 2 --data_dir [tacorl_dataset_dir]
```
> You can change "eval_type" (in_a_row or individually) and "len_task_chain" (1, 2 or 3) arguments.


## Pre-trained Models

You can download pretrained models at the following URLs:

1. Franka Kitchen: Partial-Type [https://drive.google.com/file/d/1Ade0CV35i85ahs_z_6xVZYWddHQSOAQS/view?usp=sharing]

2. Franka Kitchen: Mixed-Type [https://drive.google.com/file/d/1MwS_PUQHzoQLfJiw_jLbxTu_BcjT70at/view?usp=sharing]

3. CALVIN [https://drive.google.com/file/d/1tZe3Cxs9jPXEbZiMwHGyuCgXeFgmrLe7/view?usp=sharing]

4. CALVIN: Extension [https://drive.google.com/file/d/1csf_pVkScONRO7hDTEOVk0rT0xcmTnXO/view?usp=sharing]

> Unzip the downloaded checkpoint file to the following path: "Playbook/results/[kitchen-partial or kitchen-mixed or calvin]/"

