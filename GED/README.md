# Python Environment and Dependencies Setup

# Python GED Training Script

The command `python run_ner.py` runs a Python script for Named Entity Recognition (NER) training. The following arguments are passed to this script:

- `--model_name_or_path`: Path of the pre-trained model (ARBERTv2 in this case).
- `--do_train`: Instructs the script to train the model.
- `--per_device_train_batch_size` and `--per_device_eval_batch_size`: Specify batch size for training and evaluation.
- `--gradient_accumulation_steps`: Number of steps to accumulate gradients before performing a backward/update pass.
- `--max_seq_length`: Maximum length of a sequence (number of tokens).
- `--gradient_checkpointing`: Enables gradient checkpointing to save memory during training at the expense of computation speed.
- `--num_train_epochs`: Number of training epochs.
- `--dataset_name`: Dataset to be used for training (areta_v4 in this case).
- `--output_dir`: Directory where the output files will be saved.
- `--learning_rate`: Learning rate for the model.
- `--evaluation_strategy`: Evaluation strategy during training.
- `--save_strategy`: Strategy to use for model checkpoint saving.
- `--logging_steps`: Number of steps to log the training progress.
- `--save_total_limit`: Total limit of checkpoints to keep.
- `--seed`: Seed for generating random numbers.
- `--overwrite_output_dir`: Overwrites the output directory if it already exists.
- `--text_column` and `--label_column`: Specify columns in the dataset containing the input text and the labels.
- `--dataloader_num_workers`: Number of subprocesses to use for data loading.
- `--dataloader_pin_memory`: Pins memory for data loader.
- `--cache_dir`: Directory to cache the preprocessed datasets.
- `--load_best_model_at_end`: Loads the best model found at the end of training.
- `--metric_for_best_model`: Metric to use to compare models.
- `--fp16`: Uses 16-bit (mixed) precision instead of 32-bit.

The `run_ner.py` script likely uses a transformer-based model for Named Entity Recognition (NER) and trains it on the provided dataset using the specified settings.
