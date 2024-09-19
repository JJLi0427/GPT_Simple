import json
import logging
import torch
import numpy as np
import time
from torch import nn, optim
from tqdm import tqdm
from gpt_model import *
from torch.utils.data import DataLoader
from data_process import data_process_pipline, GPTDataSet


raw_txt_path = './dataset/train.txt'
dataset_path = './dataset/dataset.txt'
data_dict_path = './dataset/data_dict.json'

max_pos = 1800
d_model = 768  # Embedding Size
d_k = d_v = 64  # dimension of K(=Q), V
n_heads = 8  # number of heads in Multi-Head Attention
d_ff = 2048  # FeedForward dimension
n_layers = 6  # number of Encoder of Decoder Layer

CLIP = 1
batch_size = 8
epochs = 30


def train(epochs, model, data_loader, optimizer, criterion, clip, device):
    for epoch in range(1, epochs+1):
        model.train()
        start_time = time.time()
        epoch_loss = 0
        step = 0
        for dec_inputs, dec_outputs in tqdm(data_loader):
            '''
            dec_inputs: [batch_size, tgt_len]
            dec_outputs: [batch_size, tgt_len]
            '''
            optimizer.zero_grad()
            dec_inputs = dec_inputs.to(device)
            dec_outputs = dec_outputs.to(device)
            # outputs: [batch_size * tgt_len, tgt_vocab_size]
            outputs, dec_self_attns = model(dec_inputs)
            loss = criterion(outputs, dec_outputs.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            epoch_loss += loss.item()
            step += 1
        logging.info(f'Epoch: {epoch + 1:02} | Time: {time.time() - start_time:.2f}s | Loss: {epoch_loss/step:.3f}')
        torch.save(model.state_dict(), f'GPT2-{epoch}epoch.pt')


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data, word2id, id2word = data_process_pipline(raw_txt_path, dataset_path, data_dict_path)
    vocab_size = len(word2id)

    dataset = GPTDataSet(train_data, word2id, id2word)
    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.padding_batch)

    model = GPT(d_model, vocab_size, word2id, id2word).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    total_params = sum(p.numel() for p in model.parameters())
    params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'Total parameters: {total_params}, Trainable parameters: {params_trainable}')
    
    train(epochs, model, data_loader, optimizer, criterion, CLIP, device)

