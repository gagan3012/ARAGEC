# GEC Code and Documentation

This repository contains the code and documentation for the GEC project.

This readme will be split into 5 parts:
- [Gector](#gector)
- [LLMs](#llms)
- [T5s](#t5s)
- [Corruptor](#corruptor)
- [Folder Tree](#folder-tree)

Overall We import these from modules from compute canada:
```
module load python/3.10 scipy-stack gcc arrow cuda cudnn rust 
```

If your environment is not set up, you can run the following command to set up your environment:
```
virtualenv --no-download ENV
source ENV/bin/activate
pip install --no-index --upgrade pip
pip install -r requirements.txt
```

Note: This might take a while to run.

## Gector

Gector Implementation for Arabic Language

### Installation

1. Install **NVIDIA-Apex** (for using amp with deepspeed)

    ```bash
    git clone https://github.com/NVIDIA/apex
    cd apex
    pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
    ```
2. Modify the scripts in `scripts/` to fit your system as a lot of them have my configs in them.


### Preprocess Data

1. Generate edits from parallel sents

    ```bash
    sbatch scripts/prepare_data.sh
    ```

2. Train Model

    ```bash
    sbatch scripts/train.sh
    ```

3. Inference

    ```bash
    sbatch scripts/predict.sh
    ```

In the inference script you might have to modify which epoch you wanna use for the model.

## LLMs

### Installation

First Download all the models even if you are running this on cedar as these models are often unavailable.


#### LLama
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
model = AutoModelForCausalLM.from_pretrained("huggyllama/llama-7b")

model.save_pretrained("llama-7b")
tokenizer.save_pretrained("llama-7b")
```
#### Alpaca
```
1. Convert Meta's released weights into huggingface format. Follow this guide:
    https://huggingface.co/docs/transformers/main/model_doc/llama
2. Make sure you cloned the released weight diff into your local machine. The weight diff is located at:
    https://huggingface.co/tatsu-lab/alpaca-7b/tree/main
3. Run this function with the correct paths. E.g.,
    python weight_diff.py recover --path_raw <path_to_step_1_dir> --path_diff <path_to_step_2_dir> --path_tuned <path_to_store_recovered_weights>
```

#### Vicuna
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-13b-v1.3")
model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-13b-v1.3")

model.save_pretrained("vicuna-13b-v1.3")
tokenizer.save_pretrained("vicuna-13b-v1.3")
```

MBZUAI/bactrian-x-bloom-7b1-lora

MBZUAI/bactrian-x-llama-7b-lora

### Training

To do training without peft use this script:

```bash
sbatch run_jasmine.sh
```

To do training with peft use this script:

```bash
sbatch run_peft_jasmine.sh
```

### Inference

To run inference on the models use this script:

```bash 
python infer_jasmine.py
```

Note: You might have to chnage the model names to run these I didnt use a sbatch script to run these as they are pretty slow. 

To run inference on the models with peft use this script:

```bash
python generate_alpaca.py
```

## T5s

To run the T5s you can use the following scripts Note: THis has been modified to generate Test, dev and 2015 results all in one go.

```bash
sbatch run_t5.sh <model_name>
```

If you wanna run this on synthetic data use this script this script does two stage training:

```bash
sbatch run_t5_synth.sh <model_name>
```

To just generate results and not do training use: 

```bash
sbatch run_t5_test.sh <model_name>
```

To run the models as different scripts just modify the following and run them:

```bash
./sbatch_run_t5.sh 
```

## Corruptor

To run the corruptor use the following script:

Note you need to add the dataset path to the script before running it.

```bash
./sbatch_run_corruptor.sh 
```

## Folder Tree: 

```
.
├── LLMs
│   ├── bloom_train.sh
│   ├── generate_alpaca.py
│   ├── infer_jasmine.py
│   ├── run_jasmine.sh
│   ├── run_peft_jasmine.sh
│   ├── train_jasmine.py
│   └── train_peft_jasmine.py
├── T5s
│   ├── run_summarization.py
│   ├── run_t5.sh
│   ├── run_t5_synth.sh
│   ├── run_t5_test.sh
│   └── sbatch_run_t5.sh
├── corrupt
│   ├── run_corruptor.py
│   ├── run_corruptor.sh
│   └── sbatch_run_corruptor.sh
├── gector
│   ├── configs
│   │   ├── ds_config_basic.json
│   │   └── ds_config_zero1.json
│   ├── data
│   │   ├── verb-form-vocab.txt
│   │   └── vocabulary
│   │       ├── d_tags.txt
│   │       ├── labels.txt
│   │       ├── labels_ar.txt
│   │       ├── labels_zh.txt
│   │       └── non_padded_namespaces.txt
│   ├── predict.py
│   ├── scripts
│   │   ├── predict.sh
│   │   ├── prepare_data.sh
│   │   └── train.sh
│   ├── src
│   │   ├── __pycache__
│   │   │   ├── dataset.cpython-310.pyc
│   │   │   ├── model.cpython-310.pyc
│   │   │   ├── predictor.cpython-310.pyc
│   │   │   └── trainer.cpython-310.pyc
│   │   ├── dataset.py
│   │   ├── model.py
│   │   ├── predictor.py
│   │   └── trainer.py
│   ├── train.py
│   └── utils
│       ├── __pycache__
│       │   ├── common_utils.cpython-310.pyc
│       │   ├── helpers.cpython-310.pyc
│       │   └── mismatched_utils.cpython-310.pyc
│       ├── change.txt
│       ├── common_utils.py
│       ├── csv_to_edits.py
│       ├── gen_labels.py
│       ├── generate_labels.py
│       ├── helpers.py
│       ├── labels.sh
│       ├── mismatched_utils.py
│       ├── orgtext.txt
│       ├── preprocess_data.py
│       ├── segment.py
│       └── tokenization.py
├── README.md
└── reqirements.txt
```