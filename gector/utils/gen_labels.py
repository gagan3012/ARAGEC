from os import linesep
import pandas as pd
import itertools
from multiprocessing import Pool
import concurrent.futures
from tqdm import tqdm

def process_spl_chunk(spl_chunk):
    try:
        text = []
        data = []
        df_all = []
        for spli in spl_chunk:
            for i in spli:
                if i.startswith("S"):
                    text.append(i.replace("S ", "").replace(".\n", ""))
                elif i.startswith("A"):
                    data.append(i.replace('A ', '').replace(
                        " ", "|||").replace("|||0\n", ""))
            df = pd.DataFrame([sub.split("|||") for sub in data])
            lst = ['start', 'end', 'action', 'change']
            rename = {v: k for v, k in enumerate(lst)}
            df.rename(columns=rename, inplace=True)
            df['tags'] = df['action'] + "_" + df['change']
            df = df[['start', 'end', 'action', 'change']]
            texts = "".join(text).split(" ")
            df['text'] = [
                "".join(texts[int(df['start'][i]):int(df['end'][i])]) for i in range(len(df))]
            df_all.append(df)
        return pd.concat(df_all)
    except Exception as e:
        print(f"An error occurred while processing a chunk: {e}")



def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def main(file):
    try:
        with open(file) as f:
            lines = f.readlines()

        lst = lines
        w = '\n'

        spl = [list(y) for x, y in itertools.groupby(
            lst, lambda z: z == w) if not x]

        spl_chunks = list(chunks(spl, 4))

        df_chunks = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(process_spl_chunk, chunk)
                       for chunk in spl_chunks}
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing chunks"):
                df_chunks.append(future.result())

        df_chunks = [chunk for chunk in df_chunks if chunk is not None]

        df_new = pd.concat(df_chunks, ignore_index=True)

        with open('change.txt', 'a') as fp:
            for item in df_new['change']:
                fp.write("%s\n" % item)
            print('Done')

        with open('orgtext.txt', 'a') as fp:
            for item in df_new['text']:
                fp.write("%s\n" % item)
            print('Done')

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    file1 = "/lustre07/scratch/gagan30/arocr/GEC/data/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/train/QALB-2014-L1-Train.m2"
    file2 = "/lustre07/scratch/gagan30/arocr/GEC/data/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/test/QALB-2014-L1-Test.m2"
    file3 = "/lustre07/scratch/gagan30/arocr/GEC/data/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/dev/QALB-2014-L1-Dev.m2"

    for file in [file1, file2, file3]:
        main(file)
