#!/bin/bash
#SBATCH --time=1:59:59
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --account=rrg-mageed
#SBATCH --job-name=run_t5_test
#SBATCH --output=../logs/%x.out
#SBATCH --error=../logs/%x.err
#SBATCH --mail-user=gbhatia880@gmail.com
#SBATCH --mail-type=ALL

module load python/3.10 scipy-stack gcc arrow cuda cudnn 

source ~/ENV/bin/activate

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

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

python -m torch.distributed.launch --nproc_per_node=1 run_summarization.py \
  --model_name_or_path /lustre06/project/6005442/DataBank/GEC_code/results/results/$model_name \
  --do_eval \
  --do_predict \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --gradient_checkpointing True \
  --num_train_epochs 10 \
  --train_file ../data/QALB-train.csv \
  --validation_file /lustre07/scratch/gagan30/arocr/GEC/data/QALB-2014-dev.csv \
  --test_file /lustre07/scratch/gagan30/arocr/GEC/data/QALB-2015-L1-Test.csv\
  --output_dir ../results/$model_name-$type-2014dev-2015test-results \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --logging_steps 1 \
  --save_total_limit 2 \
  --seed 42 \
  --text_column $text_column \
  --summary_column $label_column \
  --source_prefix "$type: " \
  --learning_rate 1e-4 \
  --max_source_length 256 \
  --max_target_length 256 \
  --predict_with_generate \
  --dataloader_num_workers 4 \
  --dataloader_pin_memory True \
  --load_best_model_at_end \
  --cache_dir /lustre07/scratch/gagan30/arocr/cache \
  --report_to tensorboard \

if [ $type == "corrector" ]
then
  cd /lustre07/scratch/gagan30/arocr/GEC/data/QALB-0.9.1-Dec03-2021-SharedTasks/m2Scripts
  preds=../results/$model_name-$type-2014dev-2015test-results/generated_predictions_dev.txt
  m2=../data/2014/dev/QALB-2014-L1-Dev.m2
  module load python/2.7
  perl QALB-Scorer.pl $preds $m2 utf8 > ../results/$model_name-$type-2014dev-2015test-results/generated_predictions_dev_scores.txt
  preds=../results/$model_name-$type-2014dev-2015test-results/generated_predictions.txt
  m2=../data/2015/test/QALB-2015-L1-Test.m2
  perl QALB-Scorer.pl $preds $m2 utf8 > ../results/$model_name-$type-2014dev-2015test-results/generated_predictions_scores_2015.txt
fi
