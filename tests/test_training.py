import pytest
import os
import torch
from pathlib import Path

from megatron.training import pretrain

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
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda available")
@pytest.mark.parametrize("arg_values", [
    ["--num-layers",
     "2",
     "--hidden-size",
     "768",
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
     "--adam-beta1"
     "0.9",
    "--adam-beta2",
    "0.999",
     "--micro-batch-size",
     "2",
     "--train-samples",
     "100"
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
     "1"
     "--eval-interval",
     "10",
     "--data-path",
    str(Path(__file__).absolute().joinpath("resources","openwebtext_200_text")),
    "--tokenizer-type",
     "OpenGPTX-HFTokenizer",
     "--tokenizer-model",
    str(Path(__file__).absolute().joinpath("resources","tokenizer","bpe_tokenizer.json")),
    "--seed",
     "42",
     "--reset-attention-mask",
     "--reset-position-ids"
     ]
])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda available")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs to gpus to run")
@pytest.mark.parametrize("arg_values", [
    ["--num-layers", "2", "--hidden-size", "768", "--num-attention-heads", "12", "--micro-batch-size", "2", "--encoder-seq-length", "2048", "--max-position-embeddings", "2048", "--vocab-file", str(Path(__file__).parent.absolute() / "data/gpt2/gpt2-tiny-vocab.json"), "--merge-file", str(Path(__file__).parent.absolute() / "data/gpt2/gpt2-tiny-merges.txt"), "--data-path", str(Path(__file__).parent.absolute() /"data/gpt2/meg-gpt2-openwebtext_text_document"), "--split", "40,30,30", "--train-iters", "10", "--lr", "0.0001", "--min-lr", "0.00001", "--bf16", "--reset-attention-mask", "--no-masked-softmax-fusion", "--deepspeed", "--zero-stage", "0", "--deepspeed_config", str(Path(__file__).parent.absolute() /"ds_config_zero_0.json"), "--eval-iters", "-1"]
])
def test_training_debug_distributed(arg_values):
    _ = dist_launcher(
       pretrain,
       world_size=4,
       master_port=find_free_port(),
       args=arg_values
    )
