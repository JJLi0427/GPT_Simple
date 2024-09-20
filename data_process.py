import json
import logging
import os
import torch
from tqdm import tqdm
from torch.utils.data import Dataset


def save_dataset(train_txt_path, dataset_path):
    train_datas = []
    temp_data = ''
    
    with open(train_txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in tqdm(lines, desc="Making dataset"):
        if line != '\n':
            line = line.strip()
            temp_data += (line + '\t')
        else:
            train_datas.append(temp_data)
            temp_data = ''
    with open(dataset_path, 'w', encoding='utf-8') as f:
        for train_data in tqdm(train_datas, desc="Writing to file"):
            f.write(train_data + '\n')


def save_data_dict(dataset_lines, data_dict_path):
    word_count ={}
    for data in tqdm(dataset_lines, desc="Making data_dict"):
        data = data.strip().replace('\t','')
        for word in data:
            word_count.setdefault(word, 0)
            word_count[word]+=1
    
    word2id = {"<pad>":0,"<unk>":1,"<sep>":2}
    temp = {word: i + len(word2id) for i, word in enumerate(word_count.keys())}
    word2id.update(temp)
    id2word = list(word2id.keys())
    data_dict = {"word2id":word2id,"id2word":id2word}
    with open(data_dict_path, 'w', encoding='utf-8') as f:
        json.dump(data_dict, f)


def make_train_data(data):
    train_data =[]
    for data in tqdm(data, desc="Making train_data"):
        data=data.strip()
        line = [i if i!='\t' else "<sep>" for i in data]+['<sep>']
        train_data.append(line)
    return train_data


def data_process_pipline(raw_txt_path, dataset_path, data_dict_path):
    if not os.path.exists(raw_txt_path):
        logging.error(f"File not found: {raw_txt_path}")
        exit(1)

    if not os.path.exists(dataset_path):
        save_dataset(raw_txt_path, dataset_path)
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset_lines = f.readlines()

    if not os.path.exists(data_dict_path):
        save_data_dict(dataset_lines, data_dict_path)
    with open(data_dict_path, 'r', encoding='utf-8') as f:
        data_dict = json.load(f)
        word2id, id2word = data_dict['word2id'], data_dict['id2word']
        
    train_data = make_train_data(dataset_lines)
    train_data = [[word2id[word] for word in line] for line in train_data]
    
    logging.info("Data processing pipline completed!")
    return train_data, word2id, id2word



class GPTDataSet(Dataset):
    def __init__(self, data, word2id, id2word):
        self.data = data
        self.word2id = word2id
        self.id2word = id2word

    def __getitem__(self, idx):
        it = self.data[idx]
        decoder_input = it[:-1]
        decoder_output = it[1:]
        decoder_input_len = len(decoder_input)
        decoder_output_len = len(decoder_output)
        return {
            "decoder_input": decoder_input,
            "decoder_input_len": decoder_input_len,
            "decoder_output": decoder_output,
            "decoder_output_len": decoder_output_len
        }

    def __len__(self):
        return len(self.data)

    def padding_batch(self, batch):
        decoder_input_lens = [d["decoder_input_len"] for d in batch]
        decoder_output_lens = [d["decoder_output_len"] for d in batch]
        decoder_input_maxlen = max(decoder_input_lens)
        decoder_output_maxlen = max(decoder_output_lens)
        for d in batch:
            d["decoder_input"].extend(
                [self.word2id["<pad>"]] * (decoder_input_maxlen-d["decoder_input_len"])
            )
            d["decoder_output"].extend(
                [self.word2id["<pad>"]] * (decoder_output_maxlen-d["decoder_output_len"])
            )
        decoder_inputs = torch.tensor(
            [d["decoder_input"] for d in batch], 
            dtype=torch.long
        )
        decoder_outputs = torch.tensor(
            [d["decoder_output"] for d in batch], 
            dtype=torch.long
        )

        return decoder_inputs, decoder_outputs