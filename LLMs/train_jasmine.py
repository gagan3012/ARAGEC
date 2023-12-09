import os
import sys

import torch
import torch.nn as nn
#import bitsandbytes as bnb
from datasets import load_dataset
import transformers
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, TrainerCallback

from transformers import BloomForCausalLM, BloomTokenizerFast, TrainerControl, TrainerState

from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)

from huggingface_hub import login, HfFolder
import numpy as np

# login(
#     token="",  # ADD YOUR TOKEN HERE
#     add_to_git_credential=True
# )

def main():
    # optimized for RTX 3090 and A100. For larger GPUs, increase some of these?
    MICRO_BATCH_SIZE = 12   # this could actually be 5 but i like powers of 2
    BATCH_SIZE = 32
    GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
    EPOCHS = 5  # we don't always need 3 tbh
    LEARNING_RATE = 3e-4  # the Karpathy constant
    CUTOFF_LEN = 256  # 256 accounts for about 96% of the data
    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT= 0.05
    LORA_TARGET_MODULES = [
        "query_key_value", "xxx"
    ]

    VAL_SET_SIZE = 500
    base_url = "/lustre06/project/6005442/DataBank/MO-GA/NeoX/inst_ft/"
    DATA_PATH = {"train": [base_url + "Ar-alpaca/clean_ar_alpaca.json", base_url + "arP3/merged_arabic_train.jsonl"], 
                }  # Choose dataset
    # Choose output directory
    OUTPUT_DIR = "/lustre07/scratch/gagan30/arocr/NeoX/falcon_1b_alpaca"
    model_name = f'/lustre06/project/6005442/DataBank/MO-GA/Falcon/Model'

    # repository_id = "bloom-{model_name}"

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS // world_size

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map='auto',
        load_in_8bit=True,trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)

    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=[
            "query_key_value",
            "dense",
            "dense_h_to_4h",
            "dense_4h_to_h",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = prepare_model_for_int8_training(
        model, output_embedding_layer_name="embed_out")
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    data = load_dataset("json", data_files=DATA_PATH,cache_dir="/lustre07/scratch/gagan30/arocr/cache")

    def generate_prompt(data_point):
        # sorry about the formatting disaster gotta move fast
        if data_point["input_ar"]:
            return f"""
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction_ar"]}

### Input:
{data_point["input_ar"]}

### Response:
{data_point["response_ar"]}
"""
        else:
            return f"""
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction_ar"]}

### Response:
{data_point["response_ar"]}
"""


    def tokenize(prompt):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=CUTOFF_LEN + 1,
            padding="max_length",
        )
        return {
            "input_ids": result["input_ids"][:-1],
            "attention_mask": result["attention_mask"][:-1],
        }


    def generate_and_tokenize_prompt(data_point):
        # This function masks out the labels for the input,
        # so that our loss is computed only on the response.
        user_prompt = (
            (
                f"""
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction_ar"]}

### Input:
{data_point["input_ar"]}

### Response:
    """
            )
            if data_point["input_ar"]
            else (
                f"""
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction_ar"]}

### Response:"""
            )
        )
        len_user_prompt_tokens = (
            len(
                tokenizer(
                    user_prompt,
                    truncation=True,
                    max_length=CUTOFF_LEN + 1,
                    padding="max_length",
                )["input_ids"]
            )
            - 1
        )  # no eos token
        full_tokens = tokenizer(
            user_prompt + data_point["response_ar"],
            truncation=True,
            max_length=CUTOFF_LEN + 1,
            padding="max_length",
        )["input_ids"][:-1]
        return {
            "input_ids": full_tokens,
            "labels": [-100] * len_user_prompt_tokens
            + full_tokens[len_user_prompt_tokens:],
            "attention_mask": [1] * (len(full_tokens)),
        }


    if VAL_SET_SIZE > 0:
        train_val = data["train"].train_test_split(
            test_size=VAL_SET_SIZE, shuffle=True, seed=42
        )
        train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt,num_proc=4)
        val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt,num_proc=4)
    else:
        train_data = data['train'].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    print("Generating prompts...")
    print(generate_prompt(data["train"][0]))

    # val_data = data["validation"].shuffle().map(generate_and_tokenize_prompt)
    # train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
    # test_data = data["test"].shuffle().map(generate_and_tokenize_prompt)

    from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

    class SavePeftModelCallback(TrainerCallback):
        def on_save(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
        ):
            checkpoint_folder = os.path.join(
                args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
            )       

            peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
            kwargs["model"].save_pretrained(peft_model_path)

            pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
            if os.path.exists(pytorch_model_path):
                os.remove(pytorch_model_path)
            return control

    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=TrainingArguments(
            per_device_train_batch_size=MICRO_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            per_device_eval_batch_size=MICRO_BATCH_SIZE,
            warmup_steps=100,
            num_train_epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            fp16=True,
            logging_steps=1,
            evaluation_strategy="epoch" if VAL_SET_SIZE > 0 else "no",
            save_strategy="epoch",
            output_dir=OUTPUT_DIR,  # output_dir=repository_id,
            save_total_limit=2,
            load_best_model_at_end=True if VAL_SET_SIZE > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
            eval_accumulation_steps=8,
            # deepspeed="dc_config2.json",
            # bf16 = True,
            report_to="tensorboard",
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(
            tokenizer, mlm=False),
        tokenizer=tokenizer,
        callbacks=[SavePeftModelCallback],
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != 'win32':
        print("Compiling model with Torch 2...")
        model = torch.compile(model)

    # If you want to resume a training phase, please choose 'True'
    # Else choose 'False'
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model(OUTPUT_DIR)
    # trainer.evaluate()
    model.save_pretrained(OUTPUT_DIR)
    predict_results = trainer.predict(val_data)
    labels = predict_results.predictions
    # labels = np.where(labels != -100, labels,
    #                     tokenizer.pad_token_id)
    predictions = tokenizer.batch_decode(
        labels[0], skip_special_tokens=True, 
        clean_up_tokenization_spaces=True, 
        max_length=256
    )
    predictions = [pred.strip() for pred in predictions]
    output_prediction_file = os.path.join(
        OUTPUT_DIR, "generated_predictions.txt")
    with open(output_prediction_file, "w") as writer:
        writer.write("\n".join(predictions))

    # trainer.create_model_card()

    # print("\n If there's a warning about missing keys above, please disregard :)")

if __name__ == "__main__":
    main()