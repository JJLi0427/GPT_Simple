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
import torch.distributed as dist
from datetime import datetime
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


def train(
    epochs, clip, model_path,
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
        torch.save(model.state_dict(), f'{model_path}/GPT2-{epoch}epoch.pt')


@hydra.main(config_path="config", config_name="config")
def main(cfg : DictConfig) -> None:
    if os.environ.get("WORLD_SIZE") is None:
        logging.warning("'WORLD_SIZE' not set in the environment.")
        world_size = 1
    else:
        world_size = int(os.environ["WORLD_SIZE"])
    local_rank = 0
    if os.environ.get("LOCAL_RANK") is not None:
        local_rank = int(os.environ["LOCAL_RANK"])
        
    rank = 0
    if os.environ.get("RANK") is None:
        logging.warning("'RANK' is not set in the environment.")
    else:
        rank = int(os.environ["RANK"])
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    logging.info(f'Rank: {rank}, Local Rank: {local_rank}, World Size: {world_size}')
    device = local_rank
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(OmegaConf.to_yaml(cfg))
    
    train_data, word2id, id2word = data_process_pipline(
        cfg.path.raw_txt_path, 
        cfg.path.dataset_path, 
        cfg.path.data_dict_path
    )
    vocab_size = len(word2id)

    dataset = GPTDataSet(
        train_data, 
        word2id, 
        id2word
    )
    train_sampler = DistributedSampler(dataset)
    data_loader = DataLoader(
        dataset, 
        batch_size=cfg.train.batch_size, 
        collate_fn=dataset.padding_batch,
        num_workers=cfg.train.num_workers,
        sampler=train_sampler
    )

    model = GPT(
        word2id, id2word, vocab_size,
        d_model=cfg.model.d_model,
        d_k=cfg.model.d_k,
        d_v=cfg.model.d_v,
        d_ff=cfg.model.d_ff,
        n_heads=cfg.model.n_heads,
        n_layers=cfg.model.n_layers,
        max_pos=cfg.model.max_pos
    )
    model = model.to(device)
    ddp_model = DDP(
        model, 
        device_ids=[device], 
        output_device=device
    )
    criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)
    optimizer = optim.Adam(ddp_model.parameters(), lr=cfg.train.lr)
    
    total_params = sum(p.numel() for p in model.parameters())
    params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'Total parameters: {total_params}, Trainable parameters: {params_trainable}')
    
    if not os.path.exists(cfg.path.model_path):
        os.mkdir(cfg.path.model_path)
    train(
        cfg.train.epochs, cfg.train.clip, cfg.path.model_path,
        device, data_loader, ddp_model, 
        optimizer, criterion
    )


if __name__ == '__main__':
    main()
