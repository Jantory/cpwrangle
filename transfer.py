import argparse
import os
import warnings
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, T5Tokenizer

from dataset import TextDataset
from utils.utils import compute_metrics

MAX_LENGTH = 300
warnings.filterwarnings("ignore", category=DeprecationWarning)


def prepare_dataset(data_dir: str, split: str, add_prefix: bool = False, n_samples: int = -1,
                    base_model: str = 't5-base', balance: bool = False):
    tokenizer = T5Tokenizer.from_pretrained(base_model, model_max_length=MAX_LENGTH)
    dataset = TextDataset(tokenizer=tokenizer, data_dir=data_dir, split=split, add_prefix=add_prefix,
                          n_samples=n_samples, balance=balance)
    dataset.descriptions = dataset.descriptions.reset_index(drop=True)
    dataset.targets = dataset.targets.reset_index(drop=True)
    return dataset


def get_model_path(data_dir, peft_type, base_model='t5-base'):
    data_dir = str(Path(data_dir).resolve())
    dataset_name = data_dir.split('/')[-1]
    if peft_type == 'lora':
        model_path = f"models/{base_model}/{dataset_name}-{peft_type}-lr{0.001}-r{8}-la{8}"
    elif peft_type == 'finetune':
        model_path = f"models/{base_model}/{dataset_name}-{peft_type}-lr{0.0005}-r{8}-la{8}"
    else:
        model_path = f"models/{base_model}/{dataset_name}-{peft_type}-lr{0.2}-r{8}-la{8}"
    return model_path


def transfer_inference(src_data_dir, target_data_dir, peft_type, task, base_model):
    src_data_dir = str(Path(src_data_dir).resolve())
    src_dataset_name = src_data_dir.split('/')[-1]
    target_data_dir = str(Path(target_data_dir).resolve())
    target_dataset_name = target_data_dir.split('/')[-1]

    tokenizer = T5Tokenizer.from_pretrained(base_model, model_max_length=MAX_LENGTH)
    model_path = get_model_path(src_data_dir, peft_type, base_model)

    # check if model exists
    if not os.path.exists(model_path):
        print(f"{peft_type} model for {src_dataset_name} does not exist.")
        return

    if peft_type != 'finetune':
        config = PeftConfig.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
        model = PeftModel.from_pretrained(model, model_path)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.to('cuda')

    test_d = prepare_dataset(data_dir=target_data_dir, split='test', add_prefix=False, base_model=base_model)
    test_dl = DataLoader(test_d, batch_size=16, shuffle=False, collate_fn=test_d.collate_fn)

    with torch.no_grad():
        preds = []
        gts = []
        for batch in test_dl:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            output = model(**batch)
            preds += tokenizer.batch_decode(torch.argmax(output.logits, dim=-1).detach().cpu().numpy(),
                                            skip_special_tokens=True)
            gts += tokenizer.batch_decode(batch['labels'].detach().cpu().numpy(),
                                          skip_special_tokens=True)
        prec, rec, acc, f1 = compute_metrics(preds, gts, task=task)
        print(f'The preditive performance trained on {src_dataset_name} and test on {target_dataset_name} with '
              f'{peft_type} is: f1 {f1}, acc {acc}', flush=True)


if __name__ == '__main__':
    em_dns = ['data/datasets/entity_matching/structured/iTunes-Amazon',
              'data/datasets/entity_matching/structured/Beer',
              'data/datasets/entity_matching/structured/Fodors-Zagats',
              'data/datasets/entity_matching/structured/Walmart-Amazon',
              'data/datasets/entity_matching/structured/Amazon-Google',
              'data/datasets/entity_matching/structured/DBLP-ACM',
              'data/datasets/entity_matching/structured/DBLP-GoogleScholar'
            ]
    ed_dns = ['data/datasets/error_detection/Hospital']
    di_dns = ['data/datasets/data_imputation/Buy',
              'data/datasets/data_imputation/Restaurant']
    dns = em_dns + ed_dns + di_dns

    # create parser
    parser = argparse.ArgumentParser(description='Run inference on the trained model.')
    parser.add_argument('--base_model', type=str, help='The base model to use.')
    args = parser.parse_args()

    b_model = args.base_model
    if b_model != 't5-base':
        pts = ['lora', 'p-tune', 'prefix', 'prompt']
    else:
        pts = ['lora', 'p-tune', 'prefix', 'prompt', 'finetune']

    for pt in pts:
        for src_dn in dns:
            for target_dn in dns:
                if target_dn in em_dns:
                    task = 'entity_matching'
                elif target_dn in ed_dns:
                    task = 'error_detection'
                else:
                    task = 'data_imputation'
                transfer_inference(src_dn, target_dn, pt, task, b_model)
