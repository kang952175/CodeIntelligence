import torch
import sentencepiece as spm
import pandas as pd
import numpy as np
import os
from transformer import Transformer


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

os.chdir('/Users/a24/Desktop/pyskillup/CodeIntelligence/Transformer')

DATASET_PATH = '/Users/a24/Desktop/pyskillup/CodeIntelligence/Transformer/DataSet/korean-parallel-corpora/bible'

en_train = open(os.path.join(DATASET_PATH, 'bible-all.en.txt'))
en_train_content = en_train.read()

en_train_list = en_train_content.split('\n')

ko_train = open(os.path.join(DATASET_PATH, 'bible-all.kr.txt'))
ko_train_content = ko_train.read()

ko_train_list = ko_train_content.split('\n')

data = pd.DataFrame()
data['en_raw'] = en_train_list
data['ko_raw'] = ko_train_list

data = data.reset_index(drop = True)

data['en'] = data['en_raw'].apply(lambda x : x.split(' ')[1:])
data['en'] = data['en'].apply(lambda x: (' ').join(x))
data['ko'] = data['ko_raw'].apply(lambda x: x.split(' ')[1:])
data['ko'] = data['ko'].apply(lambda x:(' ').join(x))

data = data[['en','ko']]

# variable
N = 2
HIDDEN_DIM = 256
NUM_HEAD = 8
INNER_DIM = 512

PAD_IDX = 0 # 중간 빈 문장
EOS_IDX = 3 # 문장 끝

# Load model
WEIGHT_FILE = 'best_48epoch.bin'
WEIGHT_PATH = '/Users/a24/Desktop/pyskillup/CodeIntelligence/Transformer/weights'

model = Transformer(N, HIDDEN_DIM, NUM_HEAD, INNER_DIM).to(device)
model.load_state_dict(torch.load(os.path.join(WEIGHT_PATH, WEIGHT_FILE), map_location=device))
model.eval()

data_dir = '/Users/a24/Desktop/pyskillup/dataset'

SRC_MODEL_FILE = os.path.join(data_dir, 'src.model')
TRG_MODEL_FILE = os.path.join(data_dir, 'trg.model')

sp_src = spm.SentencePieceProcessor()
sp_src.Load(SRC_MODEL_FILE)
sp_trg = spm.SentencePieceProcessor()
sp_trg.Load(TRG_MODEL_FILE)

def predict(src_sentence):
    
    dec_sentence = ''
    
    enc_src = sp_src.EncodeAsIds(src_sentence)
    dec_src = []
    dec_src = np.insert(dec_src, 0, sp_trg.bos_id())
    
    enc_src = torch.Tensor(enc_src).view(1, -1).int().to(device)
    dec_src = torch.Tensor(dec_src).view(1, -1).int().to(device)
    
    last_token = None
    last_token_idx = 0
    
    while(True):
        
        enc_output = model.encoder(enc_src)
        
        dec_logits, dec_output = model.decoder(
            input = dec_src, enc_src = enc_src, enc_output = enc_output
        )
        
        last_token = dec_output[:, last_token_idx].item()
        last_token = torch.Tensor([last_token]).view(-1,1).int().to(device)
        
        dec_src = torch.cat((dec_src, last_token), dim= -1)
        
        last_token_idx = last_token_idx + 1
        
        if last_token.item() is EOS_IDX:
            break
        
    return sp_trg.Decode(dec_src.tolist())
        
indices = np.random.choice(len(data['en']), 10, replace= False)
sentences = data['en'][indices].to_list()
answers = data['ko'][indices].to_list()

for idx in range(len(sentences)):
    sentence = sentences[idx]
    print(f'en = {sentence}')
    print(f'answer = {answers[idx]}')
    print(f'ko = {predict(sentence)}')
    
    