#!/usr/bin/env python
# coding: utf-8

# Imports
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from preprocess import clean_corpora
import multiprocessing
import numpy as np
import pandas as pd
import time

def print_time(start_time, opeartion='Operation'):
    total = time.time() - start_time
    print('In total,', opeartion, 'takes:', int(total/60), 'minute and', int(total%60), 'seconds')

# Import CSV data
df_all = pd.read_csv('path\\to\\folder')
# Concatenate all texts
df_all['all_texts'] = df_all['title'] + df_all['short_description'] + df_all['need_statement'] + df_all['essay']
# Drop rows with no description
df_all = df_all.dropna(subset=['all_texts'])
print("Number of rows in data =", df_all.shape[0])
print("Number of columns in data =", df_all.shape[1])
# Clean texts
docs = clean_corpora(df_all['all_texts'])

# Define Doc2Vec models
cores = multiprocessing.cpu_count()
print(f'# of cores: {cores}')
models = [
    # PV-DBOW
    Doc2Vec(dm=0, dbow_words=1, size=300, window=8, min_count=10, iter=10, workers=cores),
    # PV-DM w/average
    Doc2Vec(dm=1, dm_mean=1, size=300, window=8, min_count=10, iter =10, workers=cores),
]
# build vocabulary
models[0].build_vocab(documents)
models[1].reset_from(models[0])
# start training
for model in models:
    start_time = time.time()
    model.train(tagged_docs, total_examples=model.corpus_count, epochs=model.iter)
    print_time(start_time, 'Training')
# inference
start_time = time.time()
docs = docs.apply(dbow.infer_vector, steps=20)
print_time(start_time, 'Inference')
# save files as csv
with open('donorschoose_doc2vec.csv', 'w', encoding="utf-8", newline='\n') as f:
    X.to_csv(f, header=True, index=False)
