models=(arat5x_large_1024_229500-corruptor-agec)
dataset=()
index=(0 1 2 3 4 5 6 7 8 9)

for model in "${models[@]}"
do
    for index in "${index[@]}"
    do
        run_name="$model-$index"
        echo $run_name
        sbatch --job-name=$run_name run_corruptor.sh $model $dataset $index
    done
done