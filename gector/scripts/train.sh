#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --account=rrg-mageed
#SBATCH --job-name=gector_train_agec
#SBATCH --output=/lustre07/scratch/gagan30/arocr/GEC/logs/%x.out
#SBATCH --error=/lustre07/scratch/gagan30/arocr/GEC/logs/%x.err
#SBATCH --mail-user=gbhatia880@gmail.com
#SBATCH --mail-type=ALL

module load python/3.10 scipy-stack gcc arrow cuda cudnn

source ~/ENV/bin/activate

# python utils/csv_to_edits.py --csv_file /lustre07/scratch/gagan30/arocr/GEC/data/AGEC_100000.csv --name AGEC_100000 

# SUBSET=(valid train)

# for i in "${SUBSET[@]}"
# do
# SOURCE="/lustre07/scratch/gagan30/arocr/GEC/data/parallel_data/QALB-2014-${i}.src"
# TARGET="/lustre07/scratch/gagan30/arocr/GEC/data/parallel_data/QALB-2014-${i}.tgt"
# OUTPUT="QALB-2014-${i}.edits"
# python utils/preprocess_data.py -s $SOURCE -t $TARGET -o $OUTPUT
# done

# for i in "${SUBSET[@]}"
# do
# SOURCE="/lustre07/scratch/gagan30/arocr/GEC/data/parallel_data/AGEC_100000-${i}.src"
# TARGET="/lustre07/scratch/gagan30/arocr/GEC/data/parallel_data/AGEC_100000-${i}.tgt"
# OUTPUT="AGEC_100000-${i}.edits"
# python utils/preprocess_data.py -s $SOURCE -t $TARGET -o $OUTPUT
# done


# python utils/generate_labels.py --vocab /lustre07/scratch/gagan30/arocr/GEC/models/ARBERTv2/vocab.txt --output data/vocabulary/labels_ar.txt

detect_vocab_path="./data/vocabulary/d_tags.txt"
correct_vocab_path="./data/vocabulary/labels_ar.txt"
train_path="QALB-2014-train.edits"
valid_path="QALB-2014-valid.edits"
config_path="configs/ds_config_basic.json"
timestamp=`date "+%Y%0m%0d_%T"`
results_dir="/lustre07/scratch/gagan30/arocr/GEC/results"
save_dir="${results_dir}/ARBERTv2_GEC_QALB_${timestamp}"
pretrained_transformer_path="/lustre07/scratch/gagan30/arocr/GEC/models/ARBERTv2"
model_dir="/lustre07/scratch/gagan30/arocr/GEC/results/ARBERTv2_GEC_V5"
mkdir -p $save_dir
cp $0 $save_dir
cp $config_path $save_dir

run_cmd="deepspeed --include localhost:0 --master_port 42997 train.py \
    --deepspeed \
    --config_path $config_path \
    --num_epochs 100 \
    --max_len 256 \
    --train_batch_size 4 \
    --accumulation_size 1 \
    --valid_batch_size 4 \
    --cold_step_count 0 \
    --lr 1e-5 \
    --cold_lr 1e-3 \
    --skip_correct 0 \
    --skip_complex 0 \
    --sub_token_mode average \
    --special_tokens_fix 1 \
    --unk2keep 0 \
    --tp_prob 1 \
    --tn_prob 1 \
    --detect_vocab_path $detect_vocab_path \
    --correct_vocab_path $correct_vocab_path \
    --do_eval \
    --train_path $train_path \
    --valid_path $valid_path \
    --use_cache 1 \
    --save_dir $save_dir \
    --pretrained_transformer_path $pretrained_transformer_path \
    --amp \
    --model_dir $model_dir \
    --ckpt_id "epoch-82"
    2>&1 | tee ${save_dir}/train-${timestamp}.log"

echo ${run_cmd}
eval ${run_cmd}