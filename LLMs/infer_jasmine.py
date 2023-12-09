from transformers import GPTNeoForCausalLM, GPT2Tokenizer, TextStreamer, pipeline, AutoModelForCausalLM
from datasets import load_dataset
import os
from tqdm import tqdm
import torch 
from fire import Fire
import pandas as pd

def run_inference(model_name):
    model_name_or_path = f"../models/{model_name}" if os.path.isdir(f"../models/{model_name}") else f"../results/{model_name}"
    output_path = f"../output/{model_name}"

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        cache_dir="/lustre07/scratch/gagan30/arocr/cache",
    )
    tokenizer = GPT2Tokenizer.from_pretrained(
        model_name_or_path,
    )

    PROMPT_DICT = {
        "prompt_input": 
            "فيما يلي أمر/ توجيه يصف مهمة مرتبطة بمدخل لتزويد النص بسياق اضافي.  يرجى صياغة ردود."
            "مناسبة  لتحقق الطلب بطريقة مناسبة و دقيقة.\n\n"
            "الأمر/ التوجيه:\n{Instruction}\n\n المدخل:\n{input}\n\n الرد:\n",
        "prompt_no_input": 
            "فيما يلي أمر/ توجيه يصف مهمة مرتبطة بمدخل لتزويد النص بسياق اضافي.  يرجى صياغة ردود."
            "مناسبة  لتحقق الطلب بطريقة مناسبة و دقيقة.\n\n"
            "### الأمر/ التوجيه:\n{Instruction}\n\n### الرد:"
        ,
    }

    def get_prompt(prompt_type, instruction, input=None):
        if prompt_type == "prompt_input":
            return PROMPT_DICT[prompt_type].format(Instruction=instruction, input=input)
        else:
            return PROMPT_DICT[prompt_type].format(Instruction=instruction)
        
    def generate_prompt(data_point):
        if data_point["input"]:
            return get_prompt("prompt_input", data_point["Instruction"], data_point["input"])
        else:
            return get_prompt("prompt_no_input", data_point["Instruction"])
        

    # dataset = load_dataset("json", data_files="../data/QALB-JASMINE-INSFINE-TEST.json", cache_dir="/lustre07/scratch/gagan30/arocr/cache", split="train")

    # dataset = dataset.map(lambda x: {"prompt": generate_prompt(x)}, desc="Generating prompts")

    # print(dataset[0])

    # dataset = dataset[:10]

    # dataset = pd.DataFrame(dataset)

    # dataset = dataset.head()

    # with open('../data/QALB-JASMINE-INSFINE-TEST-PROMPT.json', 'w', encoding='utf-8') as file:
    #     dataset.to_json(file, force_ascii=False, orient='records', lines=True)

    dataset = load_dataset("json", data_files="../data/QALB-JASMINE-INSFINE-TEST-PROMPT.json", cache_dir="/lustre07/scratch/gagan30/arocr/cache", split="train")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
        
    params = {
        "temperature":0.9,
        "top_p":0.95,
        "top_k":10,
        "do_sample":True, 
        "max_length":512, 
        "num_return_sequences":1
    }

    jasmine27 = pipeline(
        "text-generation", model=model, tokenizer=tokenizer,
        return_full_text=False, device=0,
          torch_dtype=torch.float16)
    

    def data(dataset):
        for row in dataset:
            yield row["prompt"]

    predictions = []

    for result in tqdm(
        jasmine27(
            data(dataset),
            batch_size=1,
            **params,
        ),
        total=len(dataset),
        desc="Generating outputs",
    ):
        print(result[0]["generated_text"])
        print("------------------")
        predictions.append(result[0]["generated_text"])
        

    def generate_results(data_point):
        input = data_point["prompt"]
        # streamer = TextStreamer(tokenizer)
        text = jasmine27(input, **params)[0]['generated_text']
        # text = tokenizer.batch_decode(text)[0]
        # print(text)
        # print("------------------")
        # text = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)[0]
        return {"generated_output": text}
    
    # input_sequence = dataset[0]["prompt"]

    # print(input_sequence)

    # jasmine27_base_output = jasmine27(
    #     input_sequence, **params)[0]['generated_text']
    # print(jasmine27_base_output)

    # print("---"*10)


    # def generate_results(data_point):
    #     input = tokenizer(data_point["prompt"], padding="max_length", max_length=256, truncation=True,
    #                         return_tensors="pt").input_ids.to(model.device)
    #     # streamer = TextStreamer(tokenizer)
    #     text = model.generate(
    #         input,
    #         do_sample=True,
    #         temperature=0.9,
    #         top_k=10,
    #         top_p=0.95,
    #         max_length=100,
    #         num_return_sequences=1,
    #     )
    #     text = tokenizer.batch_decode(text)[0]
    #     print(text)
    #     print("------------------")
    #     # text = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)[0]
    #     return {"generated_output": text}

    # predictions = dataset.map(generate_results, desc="Generating outputs")['generated_output']

    def remove_prompt(text):
        print(text)
        return  text.split("الرد:")[1].strip()
    
    predictions = [remove_prompt(pred) for pred in predictions]

    # os.makedirs(output_path, exist_ok=True)

    output_prediction_file = os.path.join(
        output_path, "generated_predictions_v2.txt")
    with open(output_prediction_file, "w") as writer:
        writer.write("\n".join(predictions))

if __name__ == "__main__":
    Fire(run_inference)