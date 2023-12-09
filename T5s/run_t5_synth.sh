#!/bin/bash
#SBATCH --time=23:59:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --account=rrg-mageed
#SBATCH --job-name=cagec
#SBATCH --output=../logs/%x.out
#SBATCH --error=../logs/%x.err
#SBATCH --mail-user=gbhatia880@gmail.com
#SBATCH --mail-type=ALL

module load python/3.10 scipy-stack gcc arrow cuda cudnn 

source ~/ENV/bin/activate

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=True
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")

model_name=$1
type="corrector"  

if [ $type == "corruptor" ]
then
  text_column="corrected"
  label_column="raw"
elif [ $type == "corrector" ]
then
  text_column="raw"
  label_column="corrected"
fi

dataset=$2

if [ $dataset == "corruptor-agec" ]
then
  train_file='/lustre07/scratch/gagan30/arocr/GEC/AGEC/results_arat5x_large_1024_229500-corruptor-agec_full.csv'
elif [ $dataset == "corruptor-gold" ]
then
  train_file='/lustre07/scratch/gagan30/arocr/GEC/AGEC/results_AraT5v2_large_1024_1000000-corruptor-gold-QALB_full.csv'
elif [ $dataset == "AGEC_100000" ]
then
  train_file='/lustre07/scratch/gagan30/arocr/GEC/AGEC/AGEC_100000.csv'
elif [ $dataset == "AGEC_1000000" ]
then
  train_file='/lustre07/scratch/gagan30/arocr/GEC/AGEC/AGEC_1000000.csv'
elif [ $dataset == "AGEC_200000" ]
then 
  train_file='/lustre07/scratch/gagan30/arocr/GEC/AGEC/AGEC_200000.csv'
elif [ $dataset == "AGEC_300000" ]
then 
  train_file='/lustre07/scratch/gagan30/arocr/GEC/AGEC/AGEC_300000.csv'
elif [ $dataset == "AGEC_400000" ]
then 
  train_file='/lustre07/scratch/gagan30/arocr/GEC/AGEC/AGEC_400000.csv'
elif [ $dataset == "AGEC_500000" ]
then 
  train_file='/lustre07/scratch/gagan30/arocr/GEC/AGEC/AGEC_500000.csv'
fi


python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS run_summarization.py \
  --model_name_or_path /lustre07/scratch/gagan30/arocr/GEC/models/AraT5x/AraT5v2_large_1024_1000000 \
  --do_train \
  --do_eval \
  --do_predict \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing True \
  --num_train_epochs 3 \
  --train_file $train_file \
  --validation_file ../data/QALB-2014-dev.csv \
  --test_file ../data/QALB-2014-test.csv \
  --output_dir ../results/$model_name-$type-$dataset \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --logging_steps 1 \
  --save_total_limit 2 \
  --seed 42 \
  --text_column $text_column \
  --summary_column $label_column \
  --source_prefix "$type: " \
  --max_source_length 128 \
  --max_target_length 128 \
  --predict_with_generate \
  --dataloader_num_workers 4 \
  --dataloader_pin_memory True \
  --load_best_model_at_end \
  --cache_dir /lustre07/scratch/gagan30/arocr/cache \


# if [ $type == "corrector" ]
# then
#   python evaluate_gec.py ../results/$model_name-$type-v2/generated_predictions.txt \
#    --m2 ../data/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/test/QALB-2014-L1-Test.m2 
# fi

python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS run_summarization.py \
  --model_name_or_path ../results/$model_name-$type-$dataset \
  --do_train \
  --do_eval \
  --do_predict \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing True \
  --num_train_epochs 15 \
  --train_file ../data/QALB-2014-train.csv \
  --validation_file ../data/QALB-2014-dev.csv \
  --test_file ../data/QALB-2014-test.csv \
  --output_dir ../results/$model_name-$type-$dataset-qalb \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --logging_steps 1 \
  --save_total_limit 2 \
  --seed 42 \
  --text_column $text_column \
  --summary_column $label_column \
  --source_prefix "$type: " \
  --learning_rate 1e-4 \
  --max_source_length 128 \
  --max_target_length 128 \
  --predict_with_generate \
  --dataloader_num_workers 4 \
  --dataloader_pin_memory True \
  --load_best_model_at_end \
  --cache_dir /lustre07/scratch/gagan30/arocr/cache \

# if [ $type == "corrector" ]
# then
#   python evaluate_gec.py ../results/$model_name-$type-v2-finetuning/generated_predictions.txt \
#    --m2 ../data/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/test/QALB-2014-L1-Test.m2 
# fi
