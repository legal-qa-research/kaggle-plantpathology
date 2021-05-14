import pandas as pd
from tqdm import tqdm

df = pd.read_csv('./submission.csv')

df = df.sort_values(by='image')

f = open('submission1.txt', 'w')


with open("imagenet_classes.txt", "r") as g:
    categories = [s.strip() for s in g.readlines()]

pbar = tqdm(total=len(df))

for index, row in df.iterrows():
    f.write(f'{row["image"]} - {categories[row["labels"]]}\n')
    pbar.update(1)
    # print(row['image'], row['labels'])
f.close()
