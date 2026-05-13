# Return-to-Go Is More Than a Number: Q-Guided Alignment for Return-Conditioned Supervised Learning

This repository contains the implementation for **Q-Guided Alignment for Return-Conditioned Supervised Learning**. The code builds on return-conditioned supervised learning and adds Q-guided alignment losses for training decision-transformer-style policies.

## Setup

Install the Python dependencies used by Decision Transformer / D4RL experiments, including:

- PyTorch
- Gym
- D4RL
- MuJoCo / mujoco-py dependencies required by D4RL
- NumPy, pandas, tqdm, wandb

Prepare the D4RL trajectory pickle files and pretrained critic checkpoints before training. By default, the example below expects:

```text
./data/halfcheetah-medium-v2.pkl
./saved_q_halfcheetah-medium-v2/Q_bc.pt
```

You can also pass another dataset directory through `--data_root` and another critic checkpoint through `--pretrain_q_path`.

## Pretrain Critic

```bash
python pretrain_q.py --env halfcheetah-medium-expert-v2 --q_layernorm --gamma 0.99 
```

## Run Example

```bash
SAVE_PATH=./results/
ENV=halfcheetah
Dataset=medium

python experiment.py --seed 2 \
    --env $ENV --dataset $Dataset \
    --eta2 5 --grad_norm 15 \
    --exp_name qtc --save_path $SAVE_PATH \
    --max_iters 500 --num_steps_per_iter 1000 --lr_decay --K 20 --early_epoch 200 \
    --early_stop --use_discount --infer_no_q --alg alignment-sequence \
    --alignment_function relu_min --pretrain_q_path ./saved_q_${ENV}-${Dataset}-v2/Q_bc.pt --target_rtg 10
```

The command saves checkpoints under `SAVE_PATH`. Training logs are written to the same run directory.
