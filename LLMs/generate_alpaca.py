import torch
from peft import PeftModel
import transformers
import pandas as pd
from tqdm import tqdm

from transformers import GenerationConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GPTNeoXForCausalLM, GPT2Tokenizer, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "/lustre07/scratch/gagan30/arocr/NeoX/falcon_1b_alpaca_arP3_ar",
    trust_remote_code=True
)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass

model = AutoModelForCausalLM.from_pretrained(
    "/lustre06/project/6005442/DataBank/MO-GA/NeoX/Neox_2-7B_40k",
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
model = PeftModel.from_pretrained(
    model, 
    "/lustre07/scratch/gagan30/arocr/NeoX/Neox_2-7B_40k_alpaca_arP3_ar/checkpoint-35329/adapter_model",
    device_map="auto",
    torch_dtype=torch.float16
)

# model = AutoModelForCausalLM.from_pretrained(
#     "/lustre07/scratch/gagan30/arocr/GEC/results/jasmine-27-gec",
#     load_in_8bit=True,
#     torch_dtype=torch.float16,
#     device_map="auto",
# )


def generate_prompt(instruction_ar, input_ar=None, response_ar=None):
    if input_ar:
            return f"""
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction_ar}

### Input:
{ input_ar}

### Response:
"""
    else:
            return f"""
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{ instruction_ar}

### Response:
"""

def generate_prompt_ar(instruction_ar, input_ar=None, response_ar=None):
    if input_ar:
        return f"""
أدناه تعليمات تصف مهمة، بالإضافة إلى مدخل يوفر سياقاً أكثر. اكتب رداً يكمل الطلب بشكل مناسب.

### التعليمات:
{instruction_ar}

### المدخل:
{input_ar}

### الرد:
"""
    else:
        return f"""
أدناه تعليمات تصف مهمة. اكتب رداً يكمل الطلب بشكل مناسب.

### التعليمات:
{instruction_ar}

### الرد:
"""


# model.eval()
# model = model.to(device)


def evaluate(
        instruction,
        input=None,
        output=None,
        temperature=0.9,
        top_p=0.90,
        top_k=40,
        num_beams=5,
        **kwargs,
):
    prompt = generate_prompt_ar(instruction, input, output)
    # prompt = instruction
    print(prompt)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=1.0,
        stop_sequences=["###"],
        num_beams=num_beams,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=True,
            max_new_tokens=200,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    output = output.replace("<unk>", "")
    return output

# Old testing code follows.

def new_eval(instruction):
    from transformers import GPTNeoXForCausalLM, GPT2Tokenizer, AutoTokenizer

    model = GPTNeoXForCausalLM.from_pretrained("/lustre06/project/6005442/DataBank/MO-GA/NeoX/Neox_6-7B_65k").to("cuda")
    model = PeftModel.from_pretrained(
        model, 
        "/lustre07/scratch/gagan30/arocr/NeoX/Neox_6-7B_65k_alpaca",
        device_map="auto",
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "/lustre06/project/6005442/DataBank/MO-GA/NeoX/Neox_6-7B_65k")
    prompt = generate_prompt(instruction)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    gen_tokens = model.generate(
        input_ids=input_ids, do_sample=True, temperature=0.9, max_new_tokens=50)
    gen_text = tokenizer.batch_decode(gen_tokens)[0]
    # print(gen_text)
    return gen_text

if __name__ == "__main__":
    # testing code for readme
    # df = pd.read_json("../data/QALB-JASMINE-INSFINE-TEST.json")

    # print(df.head())

    instruction = ["أعط ثلاث نصائح للبقاء بصحة جيدة.", "أدناه تعليمات تصف مهمة. اكتب ردا يكمل الطلب بشكل مناسب. التعليمات:أذكر ثلاث أعراض لمرض القلب.",
                   "أدناه تعليمات تصف مهمة.","اكتب ردا يكمل الطلب بشكل مناسب. التعليمات: عرف داء السكري."]

    for inst in instruction:
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("Response:\n", evaluate(inst))
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    # for i in tqdm(range(len(df))):
    #     instruction = df.iloc[i]["Instruction"]
    #     input = df.iloc[i]["input"]
    #     output = df.iloc[i]["output"]
    #     print("Response:", evaluate(instruction, input=input, output=output))
    #     print()
