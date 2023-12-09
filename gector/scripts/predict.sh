#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --account=rrg-mageed
#SBATCH --job-name=gector_predict
#SBATCH --output=/lustre07/scratch/gagan30/arocr/GEC/logs/%x.out
#SBATCH --error=/lustre07/scratch/gagan30/arocr/GEC/logs/%x.err
#SBATCH --mail-user=gbhatia880@gmail.com
#SBATCH --mail-type=ALL

module load python/3.10 scipy-stack gcc arrow cuda cudnn

source ~/ENV/bin/activate

deepspeed --include localhost:0 --master_port 42991 predict.py \
    --batch_size 1 \
    --iteration_count 1000 \
    --min_len 3 \
    --max_len 256 \
    --min_error_probability 0.50 \
    --additional_confidence 0.1 \
    --sub_token_mode "average" \
    --max_pieces_per_token 5 \
    --model_dir /lustre07/scratch/gagan30/arocr/GEC/results/ARBERTv2_GEC_V5/ \
    --ckpt_id "epoch-23" \
    --deepspeed_config "/lustre07/scratch/gagan30/arocr/GEC/fast-gector/configs/ds_config_basic.json" \
    --detect_vocab_path "./data/vocabulary/d_tags.txt" \
    --correct_vocab_path "./data/vocabulary/labels_ar.txt" \
    --pretrained_transformer_path "/lustre07/scratch/gagan30/arocr/GEC/models/ARBERTv2" \
    --input_path "/lustre07/scratch/gagan30/arocr/GEC/data/parallel_data/SOURCE_test.txt" \
    --out_path "/lustre07/scratch/gagan30/arocr/GEC/fast-gector/generated_predictions.txt" \
    --special_tokens_fix 1 \
    --detokenize 1 \
    --segmented 1 \
    --amp

cd /lustre07/scratch/gagan30/arocr/GEC/data/QALB-0.9.1-Dec03-2021-SharedTasks/m2Scripts/

perl QALB-Scorer.pl \
    /lustre07/scratch/gagan30/arocr/GEC/fast-gector/generated_predictions.txt \
  /lustre07/scratch/gagan30/arocr/GEC/data/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/test/QALB-2014-L1-Test.m2 utf8




