from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets

import torch
import torch.nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import omegaconf
from typing import Union, List, Tuple, Literal

class IMDBDataset(torch.utils.data.Dataset):
    def __init__(self, data_config : omegaconf.DictConfig, split : Literal['train', 'valid', 'test']):
        """
        Inputs :
            data_config : omegaconf.DictConfig{
                model_name : str
                max_len : int
                valid_size : float
            }
            split : Literal['train', 'valid', 'test']
        Outputs : None
        """
        self.split = split
        self.max_len = data_config.data.max_len
        self.tokenizer = AutoTokenizer.from_pretrained(data_config.model.name, use_fast=True)
        
        # Load IMDB dataset
        imdb = load_dataset('stanfordnlp/imdb')
        # concatenate_datasets merges two Dataset objects into one
        train_test_combined = concatenate_datasets([imdb['train'], imdb['test']])
        
        # Split into train/valid/test with 9:1:1 ratio
        # First split into train (0.8) and temp (0.2)
        train_valid = train_test_combined.train_test_split(test_size=0.2, seed=42)
        self.train_data = train_valid['train']
        
        # Then split temp (0.2) into valid (0.5 of 0.2 = 0.1) and test (0.5 of 0.2 = 0.1)
        temp = train_valid['test']
        valid_test = temp.train_test_split(test_size=0.5, seed=42)
        self.valid_data = valid_test['train']
        self.test_data = valid_test['test']
        
        # Tokenize datasets
        self.train_tokenized = self.train_data.map(
            lambda x: self.tokenizer(x['text'], truncation=True, padding='max_length', max_length=self.max_len),
            batched=True
        )
        self.valid_tokenized = self.valid_data.map(
            lambda x: self.tokenizer(x['text'], truncation=True, padding='max_length', max_length=self.max_len),
            batched=True
        )
        self.test_tokenized = self.test_data.map(
            lambda x: self.tokenizer(x['text'], truncation=True, padding='max_length', max_length=self.max_len),
            batched=True
        )
        
        # Select the appropriate dataset
        if split == 'train':
            self.data = self.train_tokenized
        elif split == 'valid':
            self.data = self.valid_tokenized
        else:  # test
            self.data = self.test_tokenized
        
        print(f">> SPLIT : {self.split} | Total Data Length : {len(self.data)}")
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> dict:
        """
        Inputs :
            idx : int
        Outputs :
            inputs : dict{
                input_ids : torch.Tensor
                token_type_ids : torch.Tensor (optional, only for BERT-like models)
                attention_mask : torch.Tensor
                label : torch.Tensor
            }
        """
        item = self.data[idx]
        result = {
            'input_ids': item['input_ids'],
            'attention_mask': item['attention_mask'],
            'label': item['label']
        }
        # Add token_type_ids only if it exists (BERT has it, ModernBERT doesn't)
        if 'token_type_ids' in item:
            result['token_type_ids'] = item['token_type_ids']
        return result

    @staticmethod
    def collate_fn(batch : List[dict]) -> dict:
        """
        Inputs :
            batch : List[dict]
        Outputs :
            data_dict : dict{
                input_ids : torch.Tensor
                token_type_ids : torch.Tensor (optional, only for BERT-like models)
                attention_mask : torch.Tensor
                label : torch.Tensor
            }
        """
        data_dict = {
            'input_ids': [],
            'attention_mask': [],
            'label': []
        }
        # Check if token_type_ids is present in any item
        has_token_type_ids = 'token_type_ids' in batch[0]
        if has_token_type_ids:
            data_dict['token_type_ids'] = []
        
        for item in batch:
            data_dict['input_ids'].append(item['input_ids'])
            data_dict['attention_mask'].append(item['attention_mask'])
            if has_token_type_ids:
                data_dict['token_type_ids'].append(item['token_type_ids'])
            data_dict['label'].append(item['label'])
        
        data_dict['input_ids'] = torch.tensor(data_dict['input_ids'])
        data_dict['attention_mask'] = torch.tensor(data_dict['attention_mask'])
        if has_token_type_ids:
            data_dict['token_type_ids'] = torch.tensor(data_dict['token_type_ids'])
        data_dict['label'] = torch.tensor(data_dict['label'])
        
        return data_dict
    
def get_dataloader(configs : omegaconf.DictConfig, split : Literal['train', 'valid', 'test']) -> torch.utils.data.DataLoader:
    """
    Output : torch.utils.data.DataLoader
    """
    dataset = IMDBDataset(configs, split)
    batch_size = configs.train.batch_size if split == 'train' else configs.eval.batch_size
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(split=='train'), collate_fn=IMDBDataset.collate_fn)
    return dataloader