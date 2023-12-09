from transformers import T5Tokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
from glob import glob
import pandas as pd
from tqdm import tqdm
import torch
from fire import Fire
import random


def generate_prompt(model, tokenizer, sentences):
    task_prefix = "corruptor: "
    # use different length sentences to test batching

    with torch.autocast("cpu"):

        inputs = tokenizer(
            [task_prefix + sentence for sentence in sentences],
            truncation=True,
            max_length=128,
            padding="max_length",
            return_tensors="pt",
        ).to("cpu")
        print("input_ids", inputs["input_ids"].shape)

        with torch.no_grad():
            output_sequences = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                do_sample=False,  # disable sampling to test if batching affects output
                max_new_tokens=128,
            )
        print("output_sequences", output_sequences.shape)

        output_data = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)

    return output_data

def flatten(l):
    return [item for sublist in l for item in sublist]

def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]

def main(model_name, data_files, index=0):
    # models = glob("../results/*-corruptor")

    model_name = "/lustre07/scratch/gagan30/arocr/GEC/results/AraT5v2_large_1024_1000000-corruptor-gold-QALB"

    # models = [model_names]
    
    # # models.remove("../results/byt5-base-corruptor")

    # results = []

    data = glob(data_files)

    dataset = load_dataset("csv", data_files=data, cache_dir="/lustre07/scratch/gagan30/arocr/cache")

    dataset = dataset.filter(lambda x: x["corrected"] != '.')

    sentences = dataset['train'].shard(num_shards=10, index=index)['corrected']

    # #divide sentences into chunks of 100
    sentences_list = list(divide_chunks(sentences, 100))

    # df = pd.DataFrame(columns=['id','original',model_names], index=range(len(sentences)))  

    #save results in dataframe
    # for model_name in tqdm(models):
    model_name = "/lustre07/scratch/gagan30/arocr/GEC/results/AraT5v2_large_1024_1000000-corruptor-gold-QALB"
    # sentence = "هذه ال7000 من المظاهرات خرجت بتنظيم من الأحزاب المعارضة ، وسلمية ، وبدون مشاكل ، ولكن لماذا حصلت مشاكل بالمرات الأخيرة ؟ وما الذي اختلف ؟ بل لأن الحكومة أرادت زرع الفتنة بين الناس وتخويفهم وإرهابهم لتفعل ما تشاء ، وتسير البلد بالديمقراطية الزائفة بل بالدكتاتورية . فمطالب الشعب لا تسمع ورميت بالمهملات ، ورأي واحد فوق رأي الجميع ."
    print(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    for sentence in tqdm(sentences_list):
        output_results = generate_prompt(model, tokenizer, sentence)
        print(output_results)
        # results.append(output_results)
    # df[model_names] = pd.Series(flatten(results))
    del model
    del tokenizer
    torch.cuda.empty_cache()

    # df['original'] = pd.Series(sentences)
    # df['id'] = pd.Series(range(len(df)))

    # df.to_csv(f'../corruptor/results_{model_names}_{index}.csv', index=False, encoding='utf-8')

    # print(df.head(10))

def introduce_errors_arabic(examples, swap_prob=0.1, remove_prob=0.05, extra_space_prob=0.1):
    text = examples["sentence"]
    words = text.split()
    error_words = []

    i = 0
    while i < len(words):
        if i < len(words) - 1 and random.random() < swap_prob:
            error_words.append(words[i + 1])
            error_words.append(words[i])
            i += 2
        elif random.random() < remove_prob:
            i += 1
        else:
            error_words.append(words[i])
            i += 1

        if random.random() < extra_space_prob:
            error_words.append("")

    error_text = " ".join(error_words)

    examples['original'] = examples["sentence"]
    examples['error'] = error_text
    return examples

if __name__ == "__main__":
    Fire(main)

    # data = glob("../data/random_stc/*.json")

    # dataset = load_dataset("json", data_files=data,
    #                        cache_dir="/lustre07/scratch/gagan30/arocr/cache")
        
    # dataset = dataset.filter(lambda x: x["sentence"] != '.')

    # dataset = dataset['train']

    # print(dataset)

    # new_column = range(len(dataset['sentence']))

    # dataset = dataset.map(introduce_errors_arabic, remove_columns=dataset.column_names)

    # dataset = dataset.add_column("id", new_column)

    # print(dataset)

    # print(dataset[0])

    # dataset = pd.DataFrame(dataset)

    # dataset = dataset.reindex(columns=['id','original','error'])

    # dataset.to_csv("../corruptor/results_error_0.csv", index=False, encoding='utf-8')







    