import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
from einops import rearrange, reduce, repeat

from tqdm import tqdm

import time
import copy
from collections import defaultdict
import joblib
import gc
import os

os.chdir('/Users/a24/Desktop/pyskillup/CodeIntelligence/Transformer/')

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# variable
N = 2
HIDDEN_DIM = 256
NUM_HEAD = 8
INNER_DIM = 512

PAD_IDX = 0 # 중간 빈 문장
EOS_IDX = 3 # 문장 끝

# dataset
data_dir = '/Users/a24/Desktop/pyskillup/CodeIntelligence/Transformer/DataSet'
src_train_path = os.path.join(data_dir, 'src_train.pkl')
src_valid_path = os.path.join(data_dir, 'src_valid.pkl')
trg_train_path = os.path.join(data_dir, 'trg_train.pkl')
trg_valid_path = os.path.join(data_dir, 'trg_valid.pkl')

src_train = joblib.load(src_train_path)
src_valid = joblib.load(src_valid_path)
trg_train = joblib.load(trg_train_path)
trg_valid = joblib.load(trg_valid_path)

# DataLoader
VOCAB_SIZE = 10000
SEQ_LEN = 60
BATCH_SIZE = 64

class MyDataset(Dataset):
    def __init__(self, src_data, trg_data):
        super().__init__()
        
        assert len(src_data) == len(trg_data)
        
        self.src_data = src_data
        self.trg_data = trg_data
        
    def __len__(self):
        return len(self.src_data)
    
    def __getitem__(self, index):
        src = self.src_data[index]
        trg_input = self.trg_data[index]
        trg_output = trg_input[1:SEQ_LEN]
        trg_output = np.pad(trg_output, (0,1), 'constant', constant_values= 0)
        # trg_output 배열의 뒤쪽에 값이 0인 패딩을 한 칸 추가하는 작업을 수행합니다. 이는 주로 시퀀스 데이터를 처리할 때, 모든 시퀀스를 동일한 길이로 맞추기 위해 사용
        
        return torch.Tensor(src).long(), torch.Tensor(trg_input).long(), torch.Tensor(trg_output).long()
    
train_dataset = MyDataset(src_train, trg_train)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True )

valid_dataset = MyDataset(src_valid, trg_valid)
valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

# Transformer : 1) Encoder/Decoder 2) Multi-Head Attention 3) Embedding 4) FFN

# 08 mask
def makeMask(tensor, option):
    if option == 'padding':
        tmp = torch.full_like(tensor, fill_value=PAD_IDX).to(device)
        
        mask = (tensor != tmp).float()
        
        mask = rearrange(mask, 'bs seq_len -> bs 1 1 seq_len')
        
    elif option == 'lookahead':
        padding_mask = makeMask(tensor, 'padding')
        padding_mask = repeat(
            padding_mask, 'bs 1 1 k_len -> bs 1 new k_len', new = padding_mask.shape[3]
        )
        
        mask = torch.ones_like(padding_mask)
        mask = torch.tril(mask)
        
        mask = mask * padding_mask
        
    return mask

# 02 Encoder
class Encoder(nn.Module):
    def __init__(self, N, hidden_dim, num_head, inner_dim, max_length = 100):
        super().__init__()
        
        self.N = N
        self.hidden_dim = hidden_dim
        self.num_head = num_head
        self.inner_dim = inner_dim
        
        self.embedding = nn.Embedding(num_embeddings=VOCAB_SIZE, embedding_dim=hidden_dim)
        self.pos_embedding = nn.Embedding(max_length, hidden_dim)
        self.enc_layers = nn.ModuleList([EncoderLayer(hidden_dim, num_head, inner_dim) for _ in range(N)])
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, input):
        batch_size = input.shape[0]
        seq_len = input.shape[1]
        
        mask = makeMask(input, option = 'padding')
        
        pos = torch.arange(0, seq_len).unsqueeze(0).repeat(batch_size, 1).to(device)
        
        # embedding layer
        output = self.dropout(self.embedding(input) + self.pos_embedding(pos))
        
        output = self.dropout(output)
        
        # N encoder layer
        for layer in self.enc_layers:
            output = layer(output, mask)
        
        return output

# 03 EncoderLayer
class EncoderLayer(nn.Module):
    def __init__(self, hiden_dim, num_head, inner_dim):
        super().__init__()
        
        self.hidden_dim = hiden_dim
        self.num_head = num_head
        self.inner_dim = inner_dim
        
        self.multiheadattention = Multiheadattention(hiden_dim, num_head)
        self.ffn = FFN(hiden_dim, inner_dim)
        self.layerNorm1 = nn.LayerNorm(hiden_dim)
        self.layerNorm2 = nn.LayerNorm(hiden_dim)
        
        self.dropout1 = nn.Dropout(p = 0.1)
        self.dropout2 = nn.Dropout(p = 0.1)
        
    def forward(self, input, mask = None):
        output = self.multiheadattention(srcQ = input, srcK = input, srcV = input, mask = mask)
        output = self.dropout1(output)
        output = output + input # residual connection
        output = self.layerNorm1(output)
        
        output_ = self.ffn(output)
        output_ = self.dropout2(output_)
        output = output_ + output
        output = self.layerNorm2(output)
        
        return output

# 04 Multiheadattention
class Multiheadattention(nn.Module):
    def __init__(self, hidden_dim, num_head):
        super().__init__()

        # embedding_dim, d_model, 512 in paper
        self.hidden_dim = hidden_dim
        # 8 in paper
        self.num_head = num_head
        # head_dim, d_key, d_query, d_value, 64 in paper (=512/8)
        self.head_dim = hidden_dim // num_head
        self.scale = torch.sqrt(torch.FloatTensor()).to(device)
        
        self.fcQ = nn.Linear(hidden_dim, hidden_dim)
        self.fcK = nn.Linear(hidden_dim, hidden_dim)
        self.fcV = nn.Linear(hidden_dim, hidden_dim)
        self.fcOut = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, srcQ, srcK, srcV, mask = None):
        
        ## scaled dot product attention
        Q = self.fcQ(srcQ)
        K = self.fcK(srcK)
        V = self.fcV(srcV)
        
        Q = rearrange(
            Q, 'bs seq_len (num_head head_dim) -> bs num_head seq_len head_dim', num_head = self.num_head
        )
        K_T = rearrange(
            K, 'bs seq_len (num_head head_dim) -> bs num_head head_dim seq_len', num_head= self.num_head
        )
        V = rearrange(
            V, 'bs seq_len (num_head head_dim) -> bs num_head seq_len head_dim', num_head = self.num_head
        )

        attention_score = torch.matmul(Q, K_T)

        if mask is not None:
            '''
            mask.shape
            if padding : (bs, 1, 1, k_len)
            if lookahead : (bs, 1, q_len, k_len)
            '''
            attention_score = torch.masked_fill(attention_score, (mask==0), -1e+4)
            
        attention_score = torch.softmax(attention_score, dim = -1)
        
        result = torch.matmul(self.dropout(attention_score), V)
        # result (bs, num_head, seq_len, head_dim)
        
        # CONCAT
        result = rearrange(result, 'bs num_head seq_len head_dim -> bs seq_len (num_head head_dim)')
        # result : (bs, seq_len, hidden_dim)
        
        # LINEAR
        result = self.fcOut(result)
        return result
    
# 05 FFN
class FFN(nn.Module):
    def __init__(self,hidden_dim, inner_dim):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.inner_dim = inner_dim
        
        self.fc1 = nn.Linear(hidden_dim, inner_dim)
        self.fc2 = nn.Linear(inner_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, input):
        output = input
        output = self.fc1(output)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.fc2(output)
        
        return output

# 06 Decoder
class Decoder(nn.Module):
    def __init__(self, N, hidden_dim, num_head, inner_dim, max_length = 100):
        super().__init__()
        
        # N : number of encoder layer repeated
        self.N = N
        self.hidden_dim = hidden_dim
        self.num_head = num_head
        self.inner_dim = inner_dim
        
        self.embeding = nn.Embedding(num_embeddings=VOCAB_SIZE, embedding_dim=hidden_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_length, hidden_dim) # Number of words to embed, word dimension
        
        self.dec_layers = nn.ModuleList([DecoderLayer(hidden_dim, num_head, inner_dim) for _ in range(N)])
        self.dropout = nn.Dropout(p=0.1)
        self.finalFc = nn.Linear(hidden_dim, VOCAB_SIZE)
        
    def forward(self, input, enc_src, enc_output):
        batch_size = input.shape[0]
        seq_len = input.shape[1]
        
        lookaheadMask = makeMask(input, option = 'lookahead')
        paddingMask = makeMask(enc_src, option = 'padding')
        
        pos = torch.arange(0, seq_len).unsqueeze(0).repeat(batch_size, 1).to(device)
        # embedding layer
        output = self.embeding(input)
        
        output = self.dropout(output + self.pos_embedding(pos))
        
        # N decoder layer
        for layer in self.dec_layers:
            output = layer(output, enc_output, paddingMask, lookaheadMask)
            
            logits = self.finalFc(output)
            output = torch.softmax(logits, dim = -1)
            
            output = torch.argmax(output, dim = -1)
            
            return logits, output
        
# 07 DecoderLayer
class DecoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_head, inner_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_head = num_head
        self.inner_dim = inner_dim
        
        self.multiheadattention1 = Multiheadattention(hidden_dim, num_head)
        self.layerNorm1 = nn.LayerNorm(hidden_dim)
        self.multiheadattention2 = Multiheadattention(hidden_dim, num_head)
        self.layerNorm2 = nn.LayerNorm(hidden_dim)
        self.ffn = FFN(hidden_dim, inner_dim)
        self.layerNorm3 = nn.LayerNorm(hidden_dim)
        
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)
        self.dropout3 = nn.Dropout(p=0.1)
        
    def forward(self, input, enc_output, paddingMask, lookaheadMask):
        
        # first multiheadattention
        output = self.multiheadattention1(input, input, input, lookaheadMask)
        output = self.dropout1(output)
        output = output + input
        output = self.layerNorm1(output)
        
        # second multiheadattention
        output_ = self.multiheadattention2(output, enc_output, enc_output, paddingMask)
        output_ = self.dropout2(output_)
        output = output_ + output
        output = self.layerNorm2(output)
        
        # Feedforward Network
        output_ = self.ffn(output)
        output_ = self.dropout3(output_)
        output = output + output_
        output = self.layerNorm3(output)
        
        return output

# 01 transformer architecture 
class Transformer(nn.Module):
    def __init__(self, N = 2, hidden_dim = 256, num_head = 8, inner_dim = 512) -> None:
        super().__init__()
        self.encoder = Encoder(N, hidden_dim, num_head, inner_dim)
        self.decoder = Decoder(N, hidden_dim, num_head, inner_dim)
        
    def forward(self, enc_src, dec_src):
        enc_output = self.encoder(enc_src)
        logits, output = self.decoder(dec_src, enc_src, enc_output)
        
        return logits, output

# model    
model = Transformer(N, HIDDEN_DIM, NUM_HEAD, INNER_DIM).to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr = 1e-4, weight_decay=0)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max= 100, eta_min=1e-5)

def criterion(logits, targets):
    return nn.CrossEntropyLoss(ignore_index=PAD_IDX)(logits.view(-1, VOCAB_SIZE), targets.view(-1))

# training
def training_one_epoch(model, optimizer, device, epoch, scheduler=None, dataloader=None, is_train=True):

    if is_train:
        assert optimizer is not None, "optimizer must be provided for training"
        model.train()
    else:
        model.eval()
    
    dataset_size = 0
    running_loss = 0
    running_accuracy = 0
    accuracy = 0
    
    bar = tqdm(enumerate(dataloader), total = len(dataloader))
    
    for step, (src, trg_input, trg_output) in bar:
        src = src.to(device)
        trg_input = trg_input.to(device)
        trg_output = trg_output.to(device)
        
        batch_size = src.shape[0]
        
        if is_train:
            logits, output = model(enc_src = src, dec_src = trg_input)
        else:
            with torch.no_grad():
                logits, output = model(enc_src = src, dec_src = trg_input)
                
        loss = criterion(logits, trg_output)
        
        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        
        running_loss += loss.item() * batch_size
        running_accuracy = np.mean(
            output.view(-1).detach().cpu().numpy() == trg_output.view(-1).detach().cpu().numpy())
        
        accuracy += running_accuracy
        
        dataset_size += batch_size
        epoch_loss = running_loss / dataset_size
        
        bar.set_postfix(
            Epoch = epoch, Loss = epoch_loss, LR = optimizer.param_groups[0]["lr"], \
                accuracy = accuracy / np.float64(step + 1)
        )
        
    accuracy /= len(dataloader)
        
    gc.collect()
        
    return epoch_loss, accuracy

# run
def run_training(
    model,
    optimizer,
    scheduler,
    device,
    num_epochs,
    metric_prefix = "",
    file_prefix = "",
    early_stopping = True,
    early_stopping_step = 10,
):
    if torch.backends.mps.is_available():
        print('[INFO] Using GPU:{}\n'.format(torch.backends.mps.is_built()))
        
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.inf
    history = defaultdict(list)
    early_stop_counter = 0
    
    for epoch in range(1, num_epochs + 1):
        gc.collect()
        
        train_epoch_loss, train_accuracy = training_one_epoch(
            model,
            optimizer,
            device,
            epoch = epoch,
            scheduler=scheduler,
            dataloader= train_dataloader
        )
        
        val_loss, val_accuracy = training_one_epoch(
            model,
            optimizer,
            device,
            epoch = epoch,
            scheduler=scheduler,
            dataloader=valid_dataloader,
            is_train = False
        )
        
        history[f"{metric_prefix}Train Loss"].append(train_epoch_loss)
        history[f"{metric_prefix}Train Accuracy"].append(train_accuracy)
        history[f"{metric_prefix}Valid Loss"].append(val_loss)
        history[f"{metric_prefix}Valid Accuracy"].append(val_accuracy)
        
        print(f"Valid Loss: {val_loss}")
        
        if val_loss <= best_loss:
            early_stop_counter = 0
            
            print(
                f"Validation Loss improved( {best_loss} ---> {val_loss} )"
            )
            
            # Update Best Loss
            best_loss = val_loss
            
            best_model_wts = copy.deepcopy(model.state_dict())
            
            # Check if 'weights' directory exists
            if not os.path.exists('weights'):
                os.makedirs('weights')
                
            PATH = "./Transformer/weights/{}epoch{:.0f}_Loss{:.4f}.bin".format(file_prefix, epoch, best_loss)
            torch.save(model.state_dict(), PATH)
            torch.save(model.state_dict(), f"./Transformer/weights/{file_prefix}best_{epoch}epoch.bin")
            
            print(f"Model Saved")
        
        elif early_stopping:
            early_stop_counter += 1
            if early_stop_counter > early_stopping_step:
                break
    
    # After all epochs, save the final model
    final_PATH = "./weights/final.bin"
    torch.save(model.state_dict(), final_PATH)
    print(f"Final model saved at {final_PATH}")
    
    end = time.time()
    time_elapsed = end - start
    print(
        "Training complete in {:.0f}h {:.0f}m {:.0f}s".format(
            time_elapsed // 3600,
            (time_elapsed % 3600) // 60,
            (time_elapsed % 3600) % 60
        )
    )
    print("Best Loss: {:.4f}".format(best_loss))
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, history

run_training(
    model= model,
    optimizer=optimizer,
    scheduler=scheduler,
    device = device,
    num_epochs=50,# 2000
    metric_prefix="",
    file_prefix="",
    early_stopping=True,
    early_stopping_step=10
)
