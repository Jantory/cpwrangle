import json
import os
import time
import warnings
from pathlib import Path
from tabulate import tabulate

import torch
from torch.utils.data import DataLoader
from peft import get_peft_model, PrefixTuningConfig, TaskType, PromptEncoderConfig, PromptTuningConfig, \
    PromptTuningInit, LoraConfig, PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, T5Tokenizer, get_linear_schedule_with_warmup

from dataset import TextDataset
from utils.utils import compute_metrics, parse_args
from memory.gpu_mem_track import MemTracker

MAX_LENGTH = 300
warnings.filterwarnings("ignore", category=DeprecationWarning)


def print_args(args):
    arg_data = [(arg, getattr(args, arg)) for arg in vars(args)]
    arg_data_transposed = list(map(list, zip(*arg_data)))

    table = tabulate(arg_data_transposed, tablefmt='grid')
    print('Parsed arguments:\n', table)


def create_peft_model(base_model: str = 't5-base', peft_type: str = 'prefix', num_virtual_tokens: int = 40,
                      **kwargs):
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
    if peft_type == 'finetune':
        pass
    elif peft_type == 'prefix':
        peft_config = PrefixTuningConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False,
                                         num_virtual_tokens=num_virtual_tokens)
        model = get_peft_model(model, peft_config)
    elif peft_type == 'prompt':
        peft_config = PromptTuningConfig(task_type=TaskType.SEQ_2_SEQ_LM, num_virtual_tokens=num_virtual_tokens,
                                         prompt_tuning_init_text=PromptTuningInit.TEXT)
        model = get_peft_model(model, peft_config)
    elif peft_type == 'p-tune':
        peft_config = PromptEncoderConfig(task_type=TaskType.SEQ_2_SEQ_LM, num_virtual_tokens=num_virtual_tokens,
                                          encoder_hidden_size=128)
        model = get_peft_model(model, peft_config)
    elif peft_type == 'lora':
        peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=kwargs['r'], lora_alpha=kwargs['la'],
                                 lora_dropout=0.1, bias="lora_only", target_modules=["q", "v"])
        model = get_peft_model(model, peft_config)
    elif peft_type == 'finetune':
        pass
    else:
        raise ValueError(f"Invalid type: {type}")
    model.print_trainable_parameters() if peft_type != 'finetune' \
        else print(f"Number of Trainable Parameters: {model.num_parameters()}", flush=True)
    return model


def prepare_dataset(data_dir: str, split: str, add_prefix: bool = False, n_samples: int = -1,
                    base_model: str = 't5-base', balance: bool = False):
    tokenizer = T5Tokenizer.from_pretrained(base_model, model_max_length=MAX_LENGTH)
    dataset = TextDataset(tokenizer=tokenizer, data_dir=data_dir, split=split, add_prefix=add_prefix,
                          n_samples=n_samples, balance=balance)
    dataset.descriptions = dataset.descriptions.reset_index(drop=True)
    dataset.targets = dataset.targets.reset_index(drop=True)
    return dataset


def train_model(args):
    base_model = args.base_model
    tokenizer = T5Tokenizer.from_pretrained(base_model, model_max_length=MAX_LENGTH)
    model = create_peft_model(base_model=base_model, peft_type=args.peft_type,
                              num_virtual_tokens=args.num_virtual_tokens, r=args.r, la=args.la)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print(f"The training will use {torch.cuda.device_count()} GPUs!", flush=True)
        model = torch.nn.DataParallel(model)
    model.to(device)

    data_dir = str(Path(args.data_dir).resolve())
    dataset_name = data_dir.split('/')[-1]

    train_d = prepare_dataset(data_dir=data_dir, split='train', add_prefix=args.add_prefix, balance=args.balance,
                              base_model=base_model)
    train_dl = DataLoader(train_d, batch_size=args.batch_size, shuffle=True, collate_fn=train_d.collate_fn)
    eval_d = prepare_dataset(data_dir=data_dir, split='validation', add_prefix=args.add_prefix,
                             base_model=base_model)
    eval_dl = DataLoader(eval_d, batch_size=args.batch_size, shuffle=False, collate_fn=eval_d.collate_fn)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                num_training_steps=len(train_dl) * args.epochs)
    best_f1 = 0.0
    best_acc = 0.0
    train_stats = {
        'train_loss': [], 'train_acc': [], 'train_f1': [], 'train_precision': [], 'train_recall': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_precision': [], 'val_recall': [],
        'train_time': []
    }

    model_saved = False
    for epoch in range(args.epochs):
        model.train()
        train_loss, val_loss = 0.0, 0.0
        train_preds, train_gts, val_preds, val_gts = [], [], [], []
        start_time = time.time()
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model(**batch)
            train_preds += tokenizer.batch_decode(torch.argmax(output.logits, dim=-1).detach().cpu().numpy(),
                                                  skip_special_tokens=True)
            train_gts += tokenizer.batch_decode(batch['labels'].detach().cpu().numpy(),
                                                skip_special_tokens=True)
            loss = output.loss.mean()
            train_loss += loss.detach().cpu().float().item()
            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        end_time = time.time()

        model.eval()
        with torch.no_grad():
            for batch in eval_dl:
                batch = {k: v.to(device) for k, v in batch.items()}
                output = model(**batch)
                val_loss += output.loss.mean().detach().cpu().float().item()
                val_preds += tokenizer.batch_decode(torch.argmax(output.logits, dim=-1).detach().cpu().numpy(),
                                                    skip_special_tokens=True)
                val_gts += tokenizer.batch_decode(batch['labels'].detach().cpu().numpy(),
                                                  skip_special_tokens=True)

            if epoch % args.save_freq == 0:
                save_data = {'prediction': val_preds, 'ground_truth': val_gts}
                save_path = f"results/{base_model}/{dataset_name}/{args.peft_type}-eval-epoch{epoch}-lr{args.lr}-r{args.r}-la{args.la}.json"
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))
                with open(save_path, 'w') as f:
                    json.dump(save_data, f, indent=4)

            train_prec, train_rec, train_acc, train_f1 = compute_metrics(train_preds, train_gts, task=args.task)
            val_prec, val_rec, val_acc, val_f1 = compute_metrics(val_preds, val_gts, task=args.task)
            train_loss /= len(train_dl)
            val_loss /= len(eval_dl)
            train_stats['train_loss'].append(train_loss)
            train_stats['train_acc'].append(train_acc)
            train_stats['train_f1'].append(train_f1)
            train_stats['train_precision'].append(train_prec)
            train_stats['train_recall'].append(train_rec)
            train_stats['val_loss'].append(val_loss)
            train_stats['val_acc'].append(val_acc)
            train_stats['val_f1'].append(val_f1)
            train_stats['val_precision'].append(val_prec)
            train_stats['val_recall'].append(val_rec)
            train_stats['train_time'].append(end_time - start_time)

            print(f"Epoch: {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
                  f"Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f} | "
                  f"Train Time: {end_time - start_time} secs", flush=True)

            if dataset_name not in ['Restaurant', 'Buy']:
                if val_f1 > best_f1:
                    best_f1 = val_f1
                    print(f"Saving model with validation F1: {best_f1} at epoch: {epoch}", flush=True)
                    model.save_pretrained(f"models/{base_model}/{dataset_name}-{args.peft_type}-lr{args.lr}-r{args.r}-la{args.la}") # for hyperparameter tuning
                    model_saved = True
            else:
                if val_acc > best_acc:
                    best_acc = val_acc
                    print(f"Saving model with validation Acc: {best_acc} at epoch: {epoch}", flush=True)
                    model.save_pretrained(f"models/{base_model}/{dataset_name}-{args.peft_type}-lr{args.lr}-r{args.r}-la{args.la}")
                    model_saved = True

    if not model_saved:
        model.save_pretrained(f"models/{base_model}/{dataset_name}-{args.peft_type}-lr{args.lr}-r{args.r}-la{args.la}")

    with open(f"results/{base_model}/{dataset_name}/train-stats-{args.peft_type}-lr{args.lr}-r{args.r}-la{args.la}.json", 'w') as f:
        json.dump(train_stats, f, indent=4)


def test(args):
    base_model = args.base_model
    data_dir = str(Path(args.data_dir).resolve())
    dataset_name = data_dir.split('/')[-1]

    tokenizer = T5Tokenizer.from_pretrained(base_model, model_max_length=MAX_LENGTH)
    model_path = f"models/{base_model}/{dataset_name}-{args.peft_type}-lr{args.lr}-r{args.r}-la{args.la}"
    # check if model exists
    if not os.path.exists(model_path):
        print(f"{args.peft_type} model does not exist.")
        print(f'The predictive performance on test data of {dataset_name} is: -')
        return

    if args.peft_type != 'finetune':
        config = PeftConfig.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
        model = PeftModel.from_pretrained(model, model_path)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print(f"The training will use {torch.cuda.device_count()} GPUs!", flush=True)
        model = torch.nn.DataParallel(model)
    model.to(device)

    test_d = prepare_dataset(data_dir=data_dir, split='test', add_prefix=args.add_prefix, base_model=base_model)
    test_dl = DataLoader(test_d, batch_size=args.batch_size, shuffle=False, collate_fn=test_d.collate_fn)
    model.eval()
    with torch.no_grad():
        save_data = {'prediction': [], 'ground_truth': [], 'metrics': {}}
        for batch in test_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model(**batch)
            pred = tokenizer.batch_decode(torch.argmax(output.logits, dim=-1).detach().cpu().numpy(),
                                          skip_special_tokens=True)
            gt = tokenizer.batch_decode(batch['labels'].detach().cpu().numpy(),
                                        skip_special_tokens=True)
            save_data['prediction'] += pred
            save_data['ground_truth'] += gt

        prec, rec, acc, f1 = compute_metrics(save_data['prediction'], save_data['ground_truth'], task=args.task)
        save_data['metrics'] = {'precision': prec, 'recall': rec, 'accuracy': acc, 'f1': f1}
        print(f'The predictive performance on test data of {dataset_name} is:', save_data['metrics'], flush=True)
        save_path = f"results/{base_model}/{dataset_name}/{args.peft_type}-test-lr{args.lr}-r{args.r}-la{args.la}.json"
        with open(save_path, 'w') as f:
            json.dump(save_data, f, indent=4)


if __name__ == '__main__':
    init_args = parse_args()
    print_args(init_args)
    train_model(init_args)
    test(init_args)
