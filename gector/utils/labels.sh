#!/bin/bash
#SBATCH --time=10:59:59
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --account=def-mageed
#SBATCH --job-name=labels
#SBATCH --output=../logs/%x.out
#SBATCH --error=../logs/%x.err
#SBATCH --mail-user=gbhatia880@gmail.com
#SBATCH --mail-type=ALL

module load python/3.10 scipy-stack gcc arrow cuda cudnn 

source ~/ENV/bin/activate

# python gen_labels.py

python utils/generate_labels.py --vocab /lustre07/scratch/gagan30/arocr/GEC/models/ARBERTv2/vocab.txt --correct_vocab utils/change.txt --org_vocab utils/orgtext.txt --output data/vocabulary/labels_ar.txt