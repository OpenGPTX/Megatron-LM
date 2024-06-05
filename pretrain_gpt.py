# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""Pretrain GPT"""

import argparse
from functools import partial

import torch

from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import get_tokenizer
from megatron.core import tensor_parallel
from megatron.core.enums import ModelType
from megatron.data.gpt_dataset import build_train_valid_test_datasets
from megatron.model import GPTModel
from megatron.training import pretrain
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.utils import average_losses_across_data_parallel_group
from megatron.arguments import core_transformer_config_from_args


import fasttext
import re
import matplotlib.pyplot as plt
from collections import Counter
import os
import seaborn as sns
import numpy as np
from collections import Counter
identified_languages_list = []

# Load the FastText language identification model
model = fasttext.load_model('/workspace/dataset/lid.176.ftz')  # Download from https://fasttext.cc/docs/en/language-identification.html

def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building GPT model ...')
    config = core_transformer_config_from_args(get_args())
    model = GPTModel(
        config,
        num_tokentypes=0,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process
    )
    return model


def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['text']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)

    return tokens, labels, loss_mask, attention_mask, position_ids

def loss_func(loss_mask, output_tensor):
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss, {'lm loss': averaged_loss[0]}


# Function to detect language using FastText
def detect_language(text):
    text = text.strip()  # Remove leading and trailing whitespace, including newline character
    prediction = model.predict(text, k=1)
    return re.search(r'__label__(\w+)', prediction[0][0]).group(1)


# Perform language identification
def identify_language(lines):
    languages = []
    for line in lines:
        prediction = model.predict(line, k=1)  # k=1 means we want the top prediction
        lang_code = prediction[0][0].replace("__label__", "")
        languages.append(lang_code)
    return languages


def plot_histogram(languages, save_path=None):
    language_counts = Counter(languages)
    total_count = sum(language_counts.values())  # Total count of all languages

    plt.figure(figsize=(12, 8), dpi=200)  # Set the figure size and DPI
    bars = plt.bar(language_counts.keys(), language_counts.values(), color='skyblue')

    # Add percentages on top of each bar
    for bar in bars:
        height = bar.get_height()
        percentage = height / total_count * 100
        plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{percentage:.2f}%', ha='center', va='bottom')

    plt.xlabel('Languages')
    plt.ylabel('Counts')
    plt.title('Language Identification Histogram')
    plt.xticks(rotation=45)
    plt.grid()
    plt.tight_layout()

    # Save the plot if save_path is provided
    if save_path:
        directory = os.path.dirname(save_path)
        # if not os.path.exists(directory):
        #     os.makedirs(directory)
        plt.savefig(save_path, dpi=200)  # Save the plot with specified DPI
        print(f"Plot saved as '{save_path}'")
    else:
        plt.show()


# Function to plot sample index versus identified language and save the plot with higher resolution
def plot_language_samples(languages, save_path=None):
    plt.figure(figsize=(12, 8), dpi=200)  # Set the figure size and DPI

    # Plot sample index versus identified language
    plt.scatter(range(len(languages)), languages, marker='.', color='skyblue')
    print("number of samples: ", len(languages))
    plt.xlabel('Sample Index')
    plt.ylabel('Identified Language')
    plt.title('Sample Index vs. Identified Language')
    plt.grid()
    plt.yticks(rotation=45)  # Rotate y-axis labels for better readability
    plt.tight_layout()

    # Save the plot if save_path is provided
    if save_path:
        directory = os.path.dirname(save_path)
        plt.savefig(save_path, dpi=200)  # Save the plot with specified DPI
        print(f"Plot saved as '{save_path}'")
    else:
        plt.show()

def plot_language_counts_barplot(languages, chunk_size=50, save_path=None):
    # Function to count 'en', 'de', and 'other'
    def count_languages(languages, chunk_size):
        chunks = [languages[i:i + chunk_size] for i in range(0, len(languages), chunk_size)]
        counts = []
        
        for chunk in chunks:
            counter = Counter(chunk)
            counts.append({
                'en': counter.get('en', 0),
                'de': counter.get('de', 0),
                'other': sum(count for lang, count in counter.items() if lang not in {'en', 'de'})
            })
        
        return counts

    # Count languages in chunks
    language_counts = count_languages(languages, chunk_size)
    language_counts = language_counts[:20]

    # Extract counts for plotting
    en_counts = [count['en'] for count in language_counts]
    de_counts = [count['de'] for count in language_counts]
    other_counts = [count['other'] for count in language_counts]
    
    # Define the x labels
    chunk_labels = [f'{i*chunk_size+1}-{(i+1)*chunk_size}' for i in range(len(language_counts))]
    
    # Plotting the data
    x = np.arange(len(chunk_labels))  # the label locations
    width = 0.2  # the width of the bars
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width, en_counts, width, label='en')
    rects2 = ax.bar(x, de_counts, width, label='de')
    rects3 = ax.bar(x + width, other_counts, width, label='other')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Count')
    ax.set_title('Language Counts per 50 Samples')
    ax.set_xticks(x)
    ax.set_xticklabels(chunk_labels, rotation=45)
    ax.legend()

    # Attach a text label above each bar in rects, displaying its height.
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()        
        
def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        data_iterator)
    timers('batch-generator').stop()

    # TODO: Decode the tokens, save it, language detection.
    tokenizer = get_tokenizer()
    tokens_list = tokens.tolist()
    tokens_list = [item for sublist in tokens_list for item in sublist]
    detokenized_words = tokenizer.detokenize(token_ids=tokens_list).split('\n')
    identified_languages = identify_language(detokenized_words)
    
    identified_languages_list.extend(identified_languages)
    
    
    output_tensor = model(tokens, position_ids, attention_mask,
                          labels=labels)

    return output_tensor, partial(loss_func, loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for GPT ...')
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        seq_length=args.seq_length,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup),
        train_data_prefix=args.train_data_path,
        valid_data_prefix=args.valid_data_path,
        test_data_prefix=args.test_data_path,
        data_cache_path=args.data_cache_path)
    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


def extra_args_provider(parser):
    parser.add_argument('--_is_gpt', default=True, help=argparse.SUPPRESS)
    return parser


if __name__ == "__main__":

    pretrain(train_valid_test_datasets_provider,
             model_provider,
             ModelType.encoder_or_decoder,
             forward_step,
             extra_args_provider=extra_args_provider,
             args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
    
    save_path = "sample_index_vs_language_plot.png"  # Specify the path where you want to save the plot
    plot_language_samples(identified_languages_list, save_path)

    save_path = "histogram.png"  # Specify the path where you want to save the plot
    plot_histogram(identified_languages_list, save_path)
    
    save_path = "box_plot.png"
    plot_language_counts_barplot(identified_languages_list, 50, save_path)
    
    
