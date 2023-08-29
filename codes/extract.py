# -*- coding: utf-8 -*-
import os
import numpy as np
np.random.seed(13)
import jsonlines


#### generate input file for training data 

f1=open('../data/training_data.txt','r')
data1=f1.readlines()
f1.close()
Frag=[]
for i in range(len(data1)):
    if i%2==0:
        label=data1[i].strip().split(':')[1]
        seq=data1[i+1].strip()
        Frag.append(seq+' '+label)
seq_length=[]
np.random.shuffle(Frag)
fout1=open('./train_and_test/train.tsv','w')
fout2=open('./train_and_test/train-nolabel.tsv','w')
for line in Frag:
    label=line.split(' ')[1]
    seq=line.split(' ')[0]
    fout1.write(label+'\t')
    
    for i in range(len(seq)):
        if i!=(len(seq)-1):
            fout1.write(seq[i]+' ')
            fout2.write(seq[i]+' ')
        else:
            fout1.write(seq[i]+'\n')
            fout2.write(seq[i]+'\n')
    seq_length.append(len(seq))
fout1.close()
fout2.close()
print('max seq length:',max(seq_length))


#### extract features for training data
os.system('python ./extract_features.py --do_lower_case=True --input_file=./train_and_test/train-nolabel.tsv --output_file=./train_and_test/train.jsonl --vocab_file=../BERT-AMP-models/BERT-Small/vocab.txt --bert_config_file=../BERT-AMP-models/BERT-Small/bert_config.json --init_checkpoint=../BERT-AMP-models/BERT-Small/bert_model.ckpt --layers=-1 --max_seq_length=512 --batch_size=16')
layer=[]

with jsonlines.open('./train_and_test/train.jsonl') as reader:
    for obj in reader:
        sample = []
        if 'features' in obj: # Check if 'features' key exists in the current object
            for feature in obj['features']:
                if 'layers' in feature: # Check if 'layers' key exists in the current feature
                    for layer_obj in feature['layers']:
                        if 'values' in layer_obj: # Check if 'values' key exists in the current layer_obj
                            values = layer_obj['values']
                            sample.extend(values)
        layer.append(sample)

with open('./train_and_test/train_features.txt', 'w') as f:
    for sample in layer:
        f.write(' '.join(map(str, sample)) + '\n')




