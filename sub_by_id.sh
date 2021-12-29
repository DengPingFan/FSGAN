id_start=0
id_len=10
for ((pred_id=${id_start};pred_id<${id_start}+${id_len};pred_id++))
do
sbatch train_ours.sh ${pred_id}
done
