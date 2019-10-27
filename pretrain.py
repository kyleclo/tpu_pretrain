
from argparse import ArgumentParser

import os

import time
import logging
import random
import numpy as np
from tqdm import tqdm

import utils  # Most of the code in adopted from https://github.com/huggingface/pytorch-transformers/blob/master/examples/lm_finetuning/finetune_on_pregenerated.py
from pytorch_transformers.modeling_auto import AutoModelWithLMHead
from pytorch_transformers.tokenization_auto import AutoTokenizer
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule

from torch.utils.data import DataLoader, RandomSampler
import torch
import torch_xla
import torch_xla_py.xla_model as tpu_xm
import torch_xla_py.data_parallel as tpu_dp

from pytorch_transformers.modeling_roberta import RobertaModel
def RobertaModel_forward(self, input_ids, token_type_ids=None, attention_mask=None, position_ids=None, head_mask=None):
    return super(RobertaModel, self).forward(input_ids, token_type_ids, attention_mask, position_ids, head_mask)
RobertaModel.forward = RobertaModel_forward  # RobertaModel.forward has a `.item()` which doesn't work nicely with TPUs


if __name__ == '__main__':
    class Args:
        pass
    args = Args()
    args.bert_model = 'roberta-base'

    parser = ArgumentParser()
    parser.add_argument('--pregenerated_data', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument("--bert_model", type=str, required=True, help="Bert pre-trained model. Either a path to the model dir or selected from list: bert-base-uncased, bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese, roberta-base, roberta-large")
    parser.add_argument("--reduce_memory", action="store_true", help="Store training data as on-disc memmaps to massively reduce memory usage")
    parser.add_argument("--epochs", type=int, required=True, help="Number of epochs to train for")
    parser.add_argument('--total_num_training_examples', type=int, required=True)
    parser.add_argument("--gradient_accumulation_steps", type=int, required=True, help="Number of gradient accumulation steps")
    parser.add_argument("--betas", nargs=2, type=float, required=True, help="tuple specifying AdamW beta weights")
    parser.add_argument("--train_batch_size", type=int, required=True, help="Total batch size for training.")
    parser.add_argument("--warmup_steps", type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_proportion", type=float, help="Linear warmup over warmup_steps.")
    parser.add_argument("--adam_epsilon", type=float, required=True, help="Epsilon for Adam optimizer.")
    parser.add_argument("--learning_rate", type=float, required=True, help="The initial learning rate for Adam.")
    parser.add_argument("--scheduler_last_epoch", type=int, default=-1)
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--log-file', type=str)
    parser.add_argument('--track_learning_rate', action='store_true', help="if true, will track learning rate in progress bar.")
    parser.add_argument('--one_tpu', action='store_true', help="Run on one tpu core for degugging. Makes it easy to use break points")
    parser.add_argument('--tpu_report', action='store_true', help="Print xla metric report")
    args = parser.parse_args()

    # logging setup
    log_format = '%(asctime)-10s: %(message)s'
    if args.log_file is not None and args.log_file != "":
        logging.basicConfig(level=logging.INFO, filename=args.log_file, filemode='w', format=log_format)
        logging.warning(f'This will get logged to file: {args.log_file}')
    else:
        logging.basicConfig(level=logging.INFO, format=log_format)

    # create output dir
    if os.path.exists(args.output_dir):
        y_or_n = input(f'Output Dir {args.output_dir} already exists.  Write to same dir? (y/n)')
        if y_or_n != 'y':
            raise Exception('Set new output dir')
    else:
        os.makedirs(args.output_dir, exist_ok=True)

    # TPU devices
    devices = tpu_xm.get_xla_supported_devices()
    if args.one_tpu:
        devices = [devices[0]]
    n_tpu = len(devices)
    logging.info(f'Found {n_tpu} TPU cores')

    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    tokenizer.save_pretrained(args.output_dir)

    # load model
    start_epoch = utils.prepare_start_epoch(args.bert_model)
    model = AutoModelWithLMHead.from_pretrained(args.bert_model)  # Only Masked Language Modeling
    logging.info(f"Saving initial checkpoint to: {args.output_dir}")
    model.save_pretrained(args.output_dir)

    # wrap model with TPU stuff
    model = tpu_dp.DataParallel(model, device_ids=devices)

    # expected total number of updates
    total_num_updates = utils.compute_num_updates_in_epoch(num_samples=args.total_num_training_examples, batch_size=args.train_batch_size, grad_accum_steps=args.gradient_accumulation_steps, n_tpu=n_tpu)

    # expected number of warmup updates
    if args.warmup_proportion is not None:
        warmup_updates = int(args.warmup_proportion * total_num_updates)
    elif args.warmup_steps is not None:
        warmup_updates = args.warmup_steps
    else:
        raise Exception('What is the warmup?? Specify either `warmup_proportion` or `warmup_steps`')


    # define callback
    def _train_one_epoch(model, loader, device, context):
        """ Called by torch_xla_py.data_parallel. This function is executed on each core of the TPU once per epoch"""

        # model parameters
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        # one optimizer and scheduler per TPU core. Both objects are saved in `context` to be reused the next epoch
        optimizer = context.getattr_or('optimizer', AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon, betas=tuple(args.betas)))
        scheduler = context.getattr_or('scheduler', WarmupLinearSchedule(optimizer, warmup_steps=warmup_updates, t_total=total_num_updates, last_epoch=args.scheduler_last_epoch))

        tr_loss = None
        tracker = tpu_xm.RateTracker()

        model.train()
        for step, batch in loader:
            input_ids, input_mask, segment_ids, lm_label_ids, _ = batch
            outputs = model(input_ids, segment_ids, input_mask, lm_label_ids)
            loss = outputs[0]
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            tracker.add(args.train_batch_size)

            tr_loss = loss * args.gradient_accumulation_steps if step == 0 else  tr_loss + loss * args.gradient_accumulation_steps
            if (step + 1) % args.gradient_accumulation_steps == 0:
                tpu_xm.optimizer_step(optimizer)
                scheduler.step()
                optimizer.zero_grad()

        # since checkpointing happens each epoch, we only need to save the scheduler state at end of each epoch
        logging.info(f'Scheduler last_epoch {scheduler.last_epoch}')

        return tr_loss.item() / step  # `.item()` requires a trip from TPU to CPU, which is very slow. Use it only once per epoch=


    # each epoch
    for epoch in range(start_epoch, args.epochs):
        # TODO: it's dumb that this class needs to re-derive which file its using. pass in directly
        # load training set corresponding to this epoch into memory
        epoch_dataset = utils.PregeneratedDataset(epoch=epoch, training_path=args.pregenerated_data, tokenizer=tokenizer, num_data_epochs=args.epochs, reduce_memory=args.reduce_memory)
        train_sampler = RandomSampler(epoch_dataset)
        train_dataloader = DataLoader(epoch_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

        num_updates = utils.compute_num_updates_in_epoch(num_samples=train_sampler.num_samples, batch_size=args.train_batch_size, grad_accum_steps=args.gradient_accumulation_steps, n_tpu=n_tpu)
        logging.info(f"""
        Training on epoch {epoch} (i.e. {train_sampler.num_samples} samples).
        With batch size {args.train_batch_size} and {args.gradient_accumulation_steps} gradient accumulation steps,
             expect {num_updates} gradient updates when split across {n_tpu} TPUs. 
        """)

        start = time.time()
        losses = model(_train_one_epoch, train_dataloader)  # calls `tpu_training_loop` multiple times, once per TPU core

        logging.info(f'Epoch {epoch} took {round(time.time() - start, 2)} seconds. Average loss: {sum(losses)/len(losses)}')
        utils.save_checkpoint(model._models[0], epoch, args.output_dir)

    if args.tpu_report:
        logging.info(torch_xla._XLAC._xla_metrics_report())
