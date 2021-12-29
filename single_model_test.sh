exp_id=0
ep=50
python test.py --checkpoints_dir checkpoints --name ckpt_${exp_id} \
--model test --dataset_mode aligned --norm batch --use_local \
--which_epoch ${ep} --results_dir ours_ep${ep}_exp${exp_id}
