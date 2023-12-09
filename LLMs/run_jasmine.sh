#!/bin/bash
#SBATCH --time=9:59:59
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=4
#SBATCH --mem=128G
#SBATCH --account=rrg-mageed
#SBATCH --job-name=run_jasmine
#SBATCH --output=../logs/%x.out
#SBATCH --error=../logs/%x.err
#SBATCH --mail-user=gbhatia880@gmail.com
#SBATCH --mail-type=ALL

module load python/3.10 scipy-stack gcc arrow cuda cudnn 

source ~/ENV/bin/activate

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3

export NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")

# deepspeed train_jasmine.py

deepspeed --num_gpus=$NUM_GPUS --num_nodes=1 train.py \
    --model_name_or_path ../models/jasmine2.7-alpaca-ft \
    --data_path ../data/QALB-train-jasmine.json \
    --eval_data_path ../data/QALB-dev-jasmine.json \
    --test_data_path ../data/QALB-test-jasmine.json \
    --output_dir ../results/jasmine-27-gec \
    --cache_dir /lustre07/scratch/gagan30/arocr/cache \
    --num_train_epochs 4 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --eval_accumulation_steps 8 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy steps \
    --gradient_checkpointing \
    --report_to="none" \
    --save_strategy "epoch" \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 100 \
    --dataloader_num_workers 4 \
    --dataloader_pin_memory True \
    --deepspeed "dc_config2.json" \
    --fp16 \

