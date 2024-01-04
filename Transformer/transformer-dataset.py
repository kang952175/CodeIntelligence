#pip install sentencepiece 
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import sentencepiece as spm
import numpy as np
import time
import joblib
import os

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# 원하는 디렉토리로 이동
os.chdir('/Users/a24/Desktop/pyskillup/CodeIntelligence/Transformer/DataSet')

# git clone 명령어 실행
os.system('git clone https://github.com/jungyeul/korean-parallel-corpora.git')

VOCAB_SIZE = 10000
SEQ_LEN = 60

# data load
DATASET_PATH = "/Users/a24/Desktop/pyskillup/CodeIntelligence/Transformer/DataSet/korean-parallel-corpora/bible"
en_train = open(os.path.join(DATASET_PATH, 'bible-all.en.txt'))
en_train_content = en_train.read()
en_train_list = en_train_content.split("\n")
ko_train = open(os.path.join(DATASET_PATH, 'bible-all.kr.txt'))
ko_train_content = ko_train.read()
ko_train_list = ko_train_content.split("\n")

data = pd.DataFrame()
data['en_raw'] = en_train_list
data['ko_raw'] = ko_train_list

data = data.reset_index(drop = True)

data['en'] = data['en_raw'].apply(lambda x: x.split(' ')[1:])
data['en'] = data['en'].apply(lambda x: (' ').join(x))
data['ko'] = data['ko_raw'].apply(lambda x: x.split(' ')[1:])
data['ko'] = data['ko'].apply(lambda x: (' ').join(x))

data = data[['en','ko']]

with open('src.txt', mode = 'w', encoding = 'utf8') as f:
    f.write('\n'.join(data['en']))
with open('trg.txt', mode='w', encoding='utf8') as f:
    f.write('\n'.join(data['ko']))

# sentencepiece
# https://github.com/google/sentencepiece/blob/master/doc/options.md
corpus = "src.txt"
prefix = "src"
spm.SentencePieceTrainer.train(
    f"--input={corpus} --model_prefix={prefix} --vocab_size={VOCAB_SIZE}" +
    " --model_type=bpe"+
    " --max_sentence_length=999999" +
    " --pad_id=0 --pad_piece=[PAD]" +
    " --unk_id=1 --unk_piece=[UNK]" +
    " --bos_id=2 --bos_piece=[BOS]" +
    " --eos_id=3 --eos_piece=[EOS]" +
    " --user_defined_symbols=[SEP],[CLS],[MASK]");

corpus = "trg.txt"
prefix = "trg"
spm.SentencePieceTrainer.train(
    f"--input={corpus} --model_prefix={prefix} --vocab_size={VOCAB_SIZE}" +
    " --model_type=bpe"+
    " --max_sentence_length=999999" +
    " --pad_id=0 --pad_piece=[PAD]" +
    " --unk_id=1 --unk_piece=[UNK]" +
    " --bos_id=2 --bos_piece=[BOS]" +
    " --eos_id=3 --eos_piece=[EOS]" +
    " --user_defined_symbols=[SEP],[CLS],[MASK]");

# to int
sp_src = spm.SentencePieceProcessor()
sp_src.Load('src.model')

for idx in range(3):
    sentence = data['en'][idx]
    print(sp_src.EncodeAsPieces(sentence))
    print(sp_src.EncodeAsIds(sentence))
    
def en_encode(tmpstr:str):
    tmpstr = np.array(sp_src.EncodeAsIds(tmpstr))
    
    if len(tmpstr) > SEQ_LEN :
        tmpstr = tmpstr[:SEQ_LEN]
        
    else:
        tmpstr = np.pad(tmpstr, (0, SEQ_LEN - len(tmpstr)), 'constant', constant_values=sp_src.pad_id())
        
    return tmpstr

src_data = data['en']

src_list = []

for idx in range(len(src_data)):
    src_list.append(en_encode(src_data[idx]))

sp_trg = spm.SentencePieceProcessor()
sp_trg.Load('trg.model')

for idx in range(3):
    sentence = data['ko'][idx]
    print(sp_trg.EncodeAsPieces(sentence))
    print(sp_trg.EncodeAsIds(sentence))

def ko_encode(tmpstr):
    tmpstr = np.array(sp_trg.EncodeAsIds(tmpstr))
    tmpstr = np.insert(tmpstr, 0, sp_trg.bos_id())
    
    if len(tmpstr) >= SEQ_LEN:
        tmpstr = tmpstr[:SEQ_LEN-1]
        tmpstr = np.pad(tmpstr, (0, 1), 'constant', constant_values=sp_trg.eos_id())
    
    else:
        tmpstr = np.pad(tmpstr, (0, 1), 'constant', constant_values=sp_trg.eos_id())
        tmpstr = np.pad(tmpstr, (0, SEQ_LEN - len(tmpstr)), 'constant', constant_values=sp_trg.pad_id())
        
    return tmpstr 

trg_data = data['ko']

trg_list = []

for idx in range(len(trg_data)):
    trg_list.append(ko_encode(trg_data[idx]))

src_train, src_valid, trg_train, trg_valid = train_test_split(src_list, trg_list, test_size=0.2, random_state=42)

joblib.dump(src_train, 'src_train.pkl')
joblib.dump(trg_train, 'trg_train.pkl')
joblib.dump(src_valid, 'src_valid.pkl')
joblib.dump(trg_valid, 'trg_valid.pkl')
joblib.dump(src_list, 'src_list.pkl')
joblib.dump(trg_list, 'trg_list.pkl')

print("You've finished creating your dataset.")