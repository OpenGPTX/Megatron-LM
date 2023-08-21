import argparse
import os
from functools import partial
from pathlib import Path

import pytest
import torch

from megatron import get_args, get_timers, get_tokenizer, print_rank_0
from megatron.arguments import core_transformer_config_from_args
from megatron.core import tensor_parallel
from megatron.core.enums import ModelType
from megatron.data.gpt_dataset import build_train_valid_test_datasets
from megatron.model import GPTModel
from megatron.training import pretrain
from megatron.utils import average_losses_across_data_parallel_group, get_ltor_masks_and_position_ids


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


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        data_iterator)
    timers('batch-generator').stop()

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

WORKER_TIMEOUT = 120


def find_free_port():
    """https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number"""
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])


def dist_init(
    run_func,
    master_port,
    local_rank,
    world_size,
    return_dict,
    *func_args,
    **func_kwargs,
):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["LOCAL_RANK"] = str(local_rank)
    # NOTE: unit tests don't support multi-node so local_rank == global rank
    os.environ["RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)

    if torch.cuda.is_available() and torch.cuda.device_count() > local_rank:
        torch.cuda.set_device(local_rank)

    run_func(return_dict=return_dict, *func_args, **func_kwargs)


def dist_launcher(run_func, world_size, master_port, *func_args, **func_kwargs):
    """Launch processes and gracefully handle failures."""
    ctx = torch.multiprocessing.get_context("spawn")
    manager = ctx.Manager()
    return_dict = manager.dict()
    # Spawn all workers on subprocesses.
    processes = []
    for local_rank in range(world_size):
        p = ctx.Process(
            target=dist_init,
            args=(
                run_func,
                master_port,
                local_rank,
                world_size,
                return_dict,
                *func_args,
            ),
            kwargs=func_kwargs,
        )
        p.start()
        processes.append(p)

    # Now loop and wait for a test to complete. The spin-wait here isn't a big
    # deal because the number of processes will be O(#GPUs) << O(#CPUs).
    any_done = False
    while not any_done:
        for p in processes:
            if not p.is_alive():
                any_done = True
                break

    # Wait for all other processes to complete
    for p in processes:
        p.join(WORKER_TIMEOUT)

    # Collect exitcodes and terminate hanging process
    failures = []
    for rank, p in enumerate(processes):
        if p.exitcode is None:
            # If it still hasn't terminated, kill it because it hung.
            p.terminate()
            if p.is_alive():
                p.kill()
            failures.append(f"Worker {rank} hung.")
        elif p.exitcode < 0:
            failures.append(f"Worker {rank} killed by signal {-p.exitcode}")
        elif p.exitcode > 0:
            failures.append(f"Worker {rank} exited with code {p.exitcode}")

    if len(failures) > 0:
        pytest.fail("\n".join(failures), pytrace=False)

    return dict(return_dict)


@pytest.mark.parametrize("arg_values", [
    ["--num-layers",
     "2",
     "--hidden-size",
     "768",
     "--num-attention-heads",
     "2",
     "--seq-length",
     "1024",
     "--max-position-embeddings",
     "1024",
     "--position-embedding-type",
     "rope",
     "--swiglu",
     "--log-num-zeros-in-grad",
     "--attention-dropout",
     "0.1",
     "--hidden-dropout",
     "0.1",
     "--weight-decay",
     "0.01",
     "--clip-grad",
     "1.0",
     "--adam-beta1",
     "0.9",
    "--adam-beta2",
    "0.999",
     "--micro-batch-size",
     "2",
     "--train-samples",
     "100",
     "--optimizer",
     "adam",
     "--dataloader-type",
     "single",
    "--init-method-std",
    "0.02",
     "--lr",
     "1e-6",
     "--lr-decay-style",
     "constant",
     "--bf16",
     "--tensor-model-parallel-size",
     "2",
     "--pipeline-model-parallel-size",
     "1",
     "--distributed-backend",
     "nccl",
     "--eval-iters",
     "1",
     "--eval-interval",
     "10",
     "--data-path",
    str(Path(__file__).absolute().joinpath("resources","openwebtext_200_text")),
    "--tokenizer-type",
     "OpenGPTX-HFTokenizer",
     "--tokenizer-model",
    str(Path(__file__).absolute().joinpath("resources","tokenizer")),
    "--seed",
     "42",
     "--reset-attention-mask",
     "--reset-position-ids"
     ]
])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda available")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs to gpus to run")
def test_training_debug_distributed(arg_values, mocker):
    # simulate command line arguments with pytest-mock's mocker fixture
    mocker.patch("sys.argv", [Path(__file__).name] + arg_values)
    _ = dist_launcher(
       pretrain,
       world_size=torch.cuda.device_count(),
       master_port=find_free_port(),
       train_valid_test_dataset_provider=train_valid_test_datasets_provider,
       model_provider=model_provider,
       model_type=ModelType.encoder_or_decoder,
       forward_step_func=forward_step,
       extra_args_provider=extra_args_provider,
       args_defaults={'tokenizer_type': 'GPT2BPETokenizer'}
    )


