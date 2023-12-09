cd /lustre06/project/6005442/DataBank/GEC_code/results/results/
models=$(ls -d *)
echo $models
cd /scratch/gagan30/arocr/GEC/code/
# for dataset in ${datasets[@]}
#   do
#   for model in ${models[@]}
#   do
#     for type in ${types[@]}
#     do
#       job_name=$dataset-$model-$type
#       echo $job_name
#       sbatch --job-name=$job_name run_t5_v2_fix.sh $model $dataset
#     done
#   done
# done

for model in ${models[@]}
do
  job_name=$model-2015
  echo $job_name
  sbatch --job-name=$job_name run_t5_test.sh $model
done