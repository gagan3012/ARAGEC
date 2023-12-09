#!/bin/bash
#SBATCH --time=23:59:59
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --account=rrg-mageed
#SBATCH --job-name=run_t5_multi
#SBATCH --output=../logs/%x.out
#SBATCH --error=../logs/%x.err
#SBATCH --mail-user=gbhatia880@gmail.com
#SBATCH --mail-type=ALL

module load python/3.10 scipy-stack gcc arrow cuda cudnn 

source ~/ENV/bin/activate

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export WANDB_DISABLED=true

model_name="AraT5v2_large_1024_1000000"
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

python run_summarization.py \
  --model_name_or_path /lustre07/scratch/gagan30/arocr/GEC/models/AraT5x/AraT5v2_large_1024_1000000 \
  --do_train \
  --do_eval \
  --do_predict \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --gradient_checkpointing True \
  --num_train_epochs 3 \
  --train_file "/lustre07/scratch/gagan30/arocr/GEC/data/AGEC training set.csv" \
  --validation_file ../data/QALB-2014-dev.csv \
  --test_file ../data/QALB-2014-test.csv \
  --output_dir ../results/AraT5v2_large_1024_1000000-agec-full \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --logging_steps 1 \
  --save_total_limit 1 \
  --seed 42 \
  --text_column $text_column \
  --summary_column $label_column \
  --source_prefix "$type: " \
  --max_source_length 128 \
  --max_target_length 128 \
  --predict_with_generate \
  --generation_num_beams 5 \
  --generation_max_length 128 \
  --dataloader_num_workers 4 \
  --dataloader_pin_memory True \
  --cache_dir /lustre07/scratch/gagan30/arocr/cache \
  --overwrite_output_dir \

# if [ $type == "corrector" ]
# then
#   python evaluate_gec.py ../results/$model_name-$type/generated_predictions.txt \
#    --m2 ../data/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/test/QALB-2014-L1-Test.m2 
# fi
