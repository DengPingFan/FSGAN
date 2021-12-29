#!/bin/bash
#SBATCH -n 10
#SBATCH --gres=gpu:v100:1
#SBATCH --time=48:00:00

exp_start=7
exp_len=10
ep_start=80
ep_len=10
for ((ep=${ep_start};ep<${ep_start}+${ep_len};ep++))
do
for ((exp_id=${exp_start};exp_id<${exp_start}+${exp_len};exp_id++))
do
python test.py --checkpoints_dir checkpoints --name ckpt_${exp_id} \
--model test --dataset_mode aligned --norm batch --use_local \
--which_epoch ${ep} --results_dir ours_ep${ep}_exp${exp_id}
done
done

nvidia-smi
hostname
