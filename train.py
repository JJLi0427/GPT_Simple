import json
import logging
import torch
import numpy as np
import time
import hydra
from omegaconf import DictConfig, OmegaConf
from torch import nn, optim
from tqdm import tqdm
from gpt_model import *
from torch.utils.data import DataLoader
from data_process import data_process_pipline, GPTDataSet


def train(
    epochs, clip, save_path,
    device, data_loader, model, 
    optimizer, criterion
):
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
        torch.save(model.state_dict(), f'{save_path}/GPT2-{epoch}epoch.pt')


@hydra.main(config_path="config", config_name="config")
def main(cfg : DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(OmegaConf.to_yaml(cfg))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data, word2id, id2word = data_process_pipline(
        cfg.path.raw_txt_path, cfg.path.dataset_path, cfg.path.data_dict_path
    )
    vocab_size = len(word2id)

    dataset = GPTDataSet(train_data, word2id, id2word)
    data_loader = DataLoader(dataset, batch_size=cfg.train.batch_size, collate_fn=dataset.padding_batch)

    model = GPT(
        word2id, id2word, vocab_size,
        d_model=int(cfg.model.d_model),
        d_k=int(cfg.model.d_k),
        d_v=int(cfg.model.d_v),
        d_ff=int(cfg.model.d_ff),
        n_heads=int(cfg.model.n_heads),
        n_layers=int(cfg.model.n_layers),
        max_pos=int(cfg.model.max_pos)
    )
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    
    total_params = sum(p.numel() for p in model.parameters())
    params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'Total parameters: {total_params}, Trainable parameters: {params_trainable}')
    
    if not os.path.exists(cfg.path.save_path):
        os.mkdir(cfg.path.save_path)
    train(
        cfg.train.epochs, cfg.train.clip, cfg.path.save_path,
        device, data_loader, model, 
        optimizer, criterion
    )


if __name__ == '__main__':
    main()
