import numpy as np
from torch.utils.data import Dataset
import torch
import random

from utils import data_utils, constants


class TextDataset(Dataset):
    def __init__(self, tokenizer, data_dir, split='train', add_prefix=False, n_samples=-1, balance=False):
        self.tokenizer = tokenizer
        pd_data_files = data_utils.read_data(
            data_dir=data_dir,
            add_prefix=add_prefix,
            max_train_samples=-1,
            sep_tok='.',
            nan_tok='nan',
            class_balanced=balance
        )

        data = pd_data_files[split]
        n_samples = len(data) if n_samples == -1 else n_samples
        descriptions = data['text'].apply(lambda x: x.strip())[:n_samples]
        targets = data['label_str'].apply(lambda x: x.rstrip('\n'))[:n_samples]

        self.descriptions = descriptions
        self.targets = targets

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, idx):
        return self.tokenizer.encode(self.descriptions[idx]), self.tokenizer.encode(self.targets[idx])

    def collate_fn(self, batch):
        max_len_data = 0
        max_len_label = 0
        for description, target in batch:
            if len(description) > max_len_data: max_len_data = len(description)
            if len(target) > max_len_label: max_len_label = len(target)

        attn_masks = []
        targets = []
        descriptions = []
        for description, target in batch:
            description.extend([self.tokenizer.pad_token_id] * (max_len_data - len(description)))
            descriptions.append(description)

            attn_mask = [int(e != self.tokenizer.pad_token_id) for e in description]
            attn_masks.append(attn_mask)

            target.extend([0] * (max_len_label - len(target)))
            targets.append(target)
        model_inputs = {'input_ids': torch.LongTensor(descriptions), 'attention_mask': torch.LongTensor(attn_masks),
                        'labels': torch.LongTensor(targets)}
        return model_inputs
