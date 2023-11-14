import argparse
import os
import time
import warnings
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torch.utils.benchmark as benchmark
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, T5Tokenizer

from dataset import TextDataset
from utils.utils import compute_metrics

MAX_LENGTH = 300
warnings.filterwarnings("ignore", category=DeprecationWarning)


def prepare_dataset(data_dir: str, split: str, add_prefix: bool = False, n_samples: int = -1,
                    base_model: str = 't5-base', balance: bool = False):
    tokenizer = T5Tokenizer.from_pretrained(base_model, model_max_length=MAX_LENGTH)
    dataset = TextDataset(tokenizer=tokenizer, data_dir=data_dir, split=split, add_prefix=add_prefix, balance=balance,
                          n_samples=n_samples)
    dataset.descriptions = dataset.descriptions.reset_index(drop=True)
    dataset.targets = dataset.targets.reset_index(drop=True)
    return dataset


def compute_token_num(dataloader):
    num_tokens = 0
    for batch in dataloader:
        input_ids = batch['input_ids']
        num_tokens += input_ids.size(0) * input_ids.size(1)
    return num_tokens


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


def prepare_model_dl(data_dir, peft_type, base_model='t5-base', n_samples=500, batch_size=16):
    data_dir = str(Path(data_dir).resolve())
    dataset_name = data_dir.split('/')[-1]

    tokenizer = T5Tokenizer.from_pretrained(base_model, model_max_length=MAX_LENGTH)
    model_path = get_model_path(data_dir, peft_type, base_model)

    # check if model exists
    if not os.path.exists(model_path):
        print(f"{peft_type} model for {dataset_name} does not exist.")
        return
    if peft_type != 'finetune':
        config = PeftConfig.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
        model = PeftModel.from_pretrained(model, model_path)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.to('cuda')

    test_d = prepare_dataset(data_dir=data_dir, split='test', add_prefix=False, n_samples=n_samples,
                             base_model=base_model)  # only use 100 samples
    test_dl = DataLoader(test_d, batch_size=batch_size, shuffle=False, collate_fn=test_d.collate_fn)
    return tokenizer, model, test_dl, test_d


@torch.no_grad()
def run_inference(tokenizer, model, test_dl):
    preds = []
    for batch in test_dl:
        batch = {k: v.to(model.device) for k, v in batch.items()}
        output = model(**batch)
        preds += tokenizer.batch_decode(torch.argmax(output.logits, dim=-1).detach().cpu().numpy(),
                                        skip_special_tokens=True)


def inference(data_dir, peft_type, task, base_model='t5-base'):
    data_dir = str(Path(data_dir).resolve())
    dataset_name = data_dir.split('/')[-1]

    tokenizer = T5Tokenizer.from_pretrained(base_model, model_max_length=MAX_LENGTH)
    model_path = get_model_path(data_dir, peft_type, base_model)

    # check if model exists
    if not os.path.exists(model_path):
        print(f"{peft_type} model for {dataset_name} does not exist.")
        return
    if peft_type != 'finetune':
        config = PeftConfig.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
        model = PeftModel.from_pretrained(model, model_path)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.to('cuda')

    test_d = prepare_dataset(data_dir=data_dir, split='test', add_prefix=False, base_model=base_model)
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
        print(f'The preditive performance on test data of {dataset_name} with {peft_type} is: f1 {f1}, acc {acc}', flush=True)


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

    # create parser
    parser = argparse.ArgumentParser(description='Run inference on the trained model.')
    parser.add_argument('--base_model', type=str, help='The base model to use.')
    args = parser.parse_args()

    b_model = args.base_model
    if b_model != 't5-large':
        pts = ['lora', 'p-tune', 'prefix', 'prompt', 'finetune']
    else:
        pts = ['lora', 'p-tune', 'prefix', 'prompt']

    # #regular inference to estimate the performance
    # for pt in pts:
    #     for em_dn in em_dns:
    #         inference(em_dn, pt, 'entity_matching', base_model=b_model)
    #
    #     for ed_dn in ed_dns:
    #         inference(ed_dn, pt, 'error_detection', base_model=b_model)
    #
    #     for di_dn in di_dns:
    #         inference(di_dn, pt, 'data_imputation', base_model=b_model)

    # regular inference to estimate the inference time
    batch_sizes = [512, 384, 256, 192, 128, 96, 64, 32, 16, 8, 4, 2, 1]
    dns = em_dns + ed_dns + di_dns
    for pt in pts:
        for dn in dns:
            for bs in batch_sizes:
                try:
                    tok, m, t_dl, t_d = prepare_model_dl(dn, pt, base_model=b_model, batch_size=bs)
                    num_tok = compute_token_num(t_dl)
                    run_inference(tok, m, t_dl) # warm up and also check executabiligy
                except:
                    continue
                else:
                    # frozen_m = torch.jit.optimize_for_inference(torch.jit.script(m.eval()))
                    timer = benchmark.Timer(stmt="run_inference(tok, m, t_dl)",
                                            setup="from __main__ import run_inference",
                                            globals={
                                                "tok": tok,
                                                "m": m,
                                                "t_dl": t_dl
                                            })

                    profile_result = timer.timeit(5)
                    print(f"Inference on {dn.split('/')[-1]} with batch size {bs}.", flush=True)
                    print(f"The batch level inference time on test data of {dn.split('/')[-1]} with {pt} is "
                          f"{profile_result.mean / (len(t_d) / bs):.8f}", flush=True)
                    print(f"The sample level inference time on test data of {dn.split('/')[-1]} with {pt} is "
                          f"{profile_result.mean / len(t_d):.8f}", flush=True)
                    print(f"The token level inference time on test data of {dn.split('/')[-1]} with {pt} is "
                          f"{profile_result.mean / num_tok:.8f}", flush=True)
                    break
