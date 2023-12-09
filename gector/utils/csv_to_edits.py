
import pandas as pd
from fire import Fire

def csv_to_edits(csv_file,name='AGEC'):
    ds = pd.read_csv(csv_file, chunksize=1000000)

    ds = pd.concat(ds, ignore_index=True)

    with open(f'/lustre07/scratch/gagan30/arocr/GEC/data/parallel_data/{name}-train.src', 'w') as fp:
        for item in ds['raw']:
            fp.write("s %s\n" % item)
        print('Done')

    with open(f'/lustre07/scratch/gagan30/arocr/GEC/data/parallel_data/{name}-train.tgt', 'w') as fp:
        for item in ds['corrected']:
            fp.write("s %s\n" % item)
        print('Done')

if __name__ == "__main__":
    Fire(csv_to_edits)
