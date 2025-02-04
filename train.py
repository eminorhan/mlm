#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=fill-mask
"""
# You can also adapt this script on your own mlm task. Pointers for this are left as comments.

import argparse
import logging
import math
import os
import random
from itertools import chain

import datasets
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    SchedulerType,
    get_scheduler,
)
from transformers.utils import send_example_telemetry
from transformers.utils.versions import require_version

logger = get_logger(__name__)

require_version("datasets>=2.14.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


DATASETS = {
    "babylm_10M": {
        "train": ["data/train_10M/childes.txt", "data/train_10M/bnc_spoken.txt", "data/train_10M/gutenberg.txt", "data/train_10M/open_subtitles.txt", "data/train_10M/simple_wiki.txt", "data/train_10M/switchboard.txt"],
        "validation": ["data/dev/childes.txt", "data/dev/bnc_spoken.txt", "data/dev/gutenberg.txt", "data/dev/open_subtitles.txt", "data/dev/simple_wiki.txt", "data/dev/switchboard.txt"]
        }, 
    "babylm_100M": {
        "train": ["data/train_100M/childes.txt", "data/train_100M/bnc_spoken.txt", "data/train_100M/gutenberg.txt", "data/train_100M/open_subtitles.txt", "data/train_100M/simple_wiki.txt", "data/train_100M/switchboard.txt"],
        "validation": ["data/dev/childes.txt", "data/dev/bnc_spoken.txt", "data/dev/gutenberg.txt", "data/dev/open_subtitles.txt", "data/dev/simple_wiki.txt", "data/dev/switchboard.txt"]
        }, 
    "wikipedia_10M_1": ["eminorhan/wikipedia", "10M_1"],
    "wikipedia_10M_2": ["eminorhan/wikipedia", "10M_2"],
    "wikipedia_10M_3": ["eminorhan/wikipedia", "10M_3"],
    "wikipedia_100M_1": ["eminorhan/wikipedia", "100M_1"],
    "wikipedia_100M_2": ["eminorhan/wikipedia", "100M_2"],
    "wikipedia_100M_3": ["eminorhan/wikipedia", "100M_3"],
    "gutenberg_10M_1": ["eminorhan/gutenberg_en_dec24", "10M_1"],
    "gutenberg_10M_2": ["eminorhan/gutenberg_en_dec24", "10M_2"],
    "gutenberg_10M_3": ["eminorhan/gutenberg_en_dec24", "10M_3"],
    "gutenberg_100M_1": ["eminorhan/gutenberg_en_dec24", "100M_1"],
    "gutenberg_100M_2": ["eminorhan/gutenberg_en_dec24", "100M_2"],
    "gutenberg_100M_3": ["eminorhan/gutenberg_en_dec24", "100M_3"],
    "tinystories_10M_1": ["eminorhan/tinystories", "10M_1"],
    "tinystories_10M_2": ["eminorhan/tinystories", "10M_2"],
    "tinystories_10M_3": ["eminorhan/tinystories", "10M_3"],
    "tinystories_100M_1": ["eminorhan/tinystories", "100M_1"],
    "tinystories_100M_2": ["eminorhan/tinystories", "100M_2"],
    "tinystories_100M_3": ["eminorhan/tinystories", "100M_3"],
    "pythonedu_10M_1": ["eminorhan/python-edu", "10M_1"],
    "pythonedu_10M_2": ["eminorhan/python-edu", "10M_2"],
    "pythonedu_10M_3": ["eminorhan/python-edu", "10M_3"],
    "pythonedu_100M_1": ["eminorhan/python-edu", "100M_1"],
    "pythonedu_100M_2": ["eminorhan/python-edu", "100M_2"],
    "pythonedu_100M_3": ["eminorhan/python-edu", "100M_3"],
}


def check_file_extensions(file_list):
    # Extract the extension from the first file
    extension = file_list[0].split('.')[-1]
    
    # Check that all files have one of the allowed extensions
    for file_name in file_list:
        assert file_name.endswith(('.csv', '.json', '.txt')), "Invalid file extension."
    
    # Check that all files have the same extension
    for file_name in file_list:
        assert file_name.split('.')[-1] == extension, "Files have different extensions."

    print("All files have the same extension: {}.".format(extension))


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a Masked Language Modeling task")
    parser.add_argument("--dataset_name", type=str, help="dataset", choices=DATASETS.keys())
    parser.add_argument("--model_name_or_path", type=str, help="Path to pretrained model or model identifier from huggingface.co/models.", required=False)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="linear", help="The scheduler type to use.", choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
    parser.add_argument("--num_warmup_steps", type=int, default=100, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--model_type", type=str, default=None, help="Model type to use if training from scratch.", choices=MODEL_TYPES)
    parser.add_argument("--max_seq_length", type=int, default=None, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated.")
    parser.add_argument("--preprocessing_num_workers", type=int, default=None, help="The number of processes to use for the preprocessing.")
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--no_keep_linebreaks", action="store_true", help="Do not keep line breaks when using TXT files.")
    parser.add_argument("--mlm_probability", type=float, default=0.3, help="Ratio of tokens to mask for masked language modeling loss")
    parser.add_argument("--checkpointing_steps", type=str, default=None, help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="If the training should continue from a checkpoint folder.")
    parser.add_argument("--use_pretrained_weights", action="store_true", help="Whether to use pretrained weights.")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_mlm_no_trainer", args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # load data
    if args.dataset_name.startswith("babylm"):
        data_files = DATASETS[args.dataset_name]

        # Sanity check for file extensions
        check_file_extensions(data_files["train"])
        check_file_extensions(data_files["validation"])

        dataset_args = {"keep_linebreaks": not args.no_keep_linebreaks}
        raw_datasets = load_dataset("text", data_files=data_files, **dataset_args)
    else:
        repo_info = DATASETS[args.dataset_name]
        raw_datasets = load_dataset(repo_info[0], repo_info[1])

    # load config, model, tokenizer, etc.
    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True, token=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True, trust_remote_code=True)

    if args.model_name_or_path and args.use_pretrained_weights:
        logger.info("Loading pretrained weights")
        model = AutoModelForMaskedLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.bfloat16, from_tf=bool(".ckpt" in args.model_name_or_path), config=config, trust_remote_code=True)
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForMaskedLM.from_config(config, torch_dtype=torch.bfloat16, trust_remote_code=True)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    if args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning("The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can override this default with `--block_size xxx`.")
            max_seq_length = 1024
    else:
        if args.max_seq_length > tokenizer.model_max_length:
            logger.warning(f"The max_seq_length passed ({args.max_seq_length}) is larger than the maximum length for the model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}.")
        max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

    logger.info(f"Max seq length: {max_seq_length}")

    # we tokenize every text, then concatenate them together before splitting them in smaller parts.
    # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
    # efficient when it receives the `special_tokens_mask`.
    def tokenize_function(examples):
        return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

    with accelerator.main_process_first():
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on every text in dataset",
        )

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of
    # max_seq_length.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < max_seq_length  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // max_seq_length) * max_seq_length
        # Split by chunks of max_len.
        result = {k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)] for k, t in concatenated_examples.items()}
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
    # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
    # might be slower to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/process#map

    with accelerator.main_process_first():
        tokenized_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            desc=f"Grouping texts in chunks of {max_seq_length}",
        )

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    # Conditional for small test subsets
    if len(train_dataset) > 3:
        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Data collator
    # This one will take care of randomly masking the tokens.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=args.mlm_probability)

    # DataLoaders creation:
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_train_batch_size)  # same per device batch size as eval

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)


    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps
        if overrode_max_train_steps
        else args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, train_dataloader, eval_dataloader, lr_scheduler)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    # initialize loss counters
    train_loss = 0
    val_loss = 0

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                # keep track of the train loss
                train_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.set_description(f"lr: {lr_scheduler.scheduler.get_last_lr()[0]}")
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0 and completed_steps != 0 and step % args.gradient_accumulation_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)

                    # save model and tokenizer
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save)
                    if accelerator.is_main_process:
                        tokenizer.save_pretrained(output_dir)

                    # log train_loss & re-initialize
                    logger.info(f"Mean train loss: {train_loss.item() / (checkpointing_steps * args.gradient_accumulation_steps)}")
                    train_loss = 0

                    # save val losses 
                    model.eval()
                    for step, batch in enumerate(eval_dataloader):
                        with torch.no_grad():
                            outputs = model(**batch)
                            loss = outputs.loss
                            # keep track of the val loss
                            val_loss += loss.detach().float()

                    # log val & re-initialize
                    logger.info(f"Mean val loss: {val_loss.item() / len(eval_dataloader)}")
                    val_loss = 0

            if completed_steps >= args.max_train_steps:
                break


if __name__ == "__main__":
    main()
