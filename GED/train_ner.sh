#!/bin/bash
#SBATCH --time=10:59:59
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --account=rrg-mageed
#SBATCH --job-name=run_ner_areta
#SBATCH --output=../logs/%x.out
#SBATCH --error=../logs/%x.err
#SBATCH --mail-user=gbhatia880@gmail.com
#SBATCH --mail-type=ALL

module load python/3.10 scipy-stack gcc arrow cuda cudnn 

source ~/ENV/bin/activate

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

text_column="text"
label_column="7_tags"
model_name="ARBERTv2"

python run_ner.py \
  --model_name_or_path /lustre07/scratch/gagan30/arocr/GEC/models/$model_name \
  --do_train \
  --per_device_train_batch_size 64 \
  --per_device_eval_batch_size 64 \
  --gradient_accumulation_steps 64 \
  --max_seq_length 128 \
  --gradient_checkpointing True \
  --num_train_epochs 25 \
  --dataset_name areta_v4 \
  --output_dir ../results/$model_name-ged-7_tags \
  --learning_rate 5e-5 \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --logging_steps 1 \
  --save_total_limit 1 \
  --seed 42 \
  --overwrite_output_dir \
  --text_column $text_column \
  --label_column $label_column \
  --dataloader_num_workers 4 \
  --dataloader_pin_memory True \
  --cache_dir /lustre07/scratch/gagan30/arocr/cache \
  --load_best_model_at_end True \
  --metric_for_best_model f1 \
  --fp16 True \