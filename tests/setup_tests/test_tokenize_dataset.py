
from pathlib import Path

import pytest

from tools.preprocess_data import main


@pytest.mark.parametrize("arg_values", [
[
    "--input", str(Path(__file__).absolute().parent.parent.joinpath("resources", "tokenization/oscar_1800.jsonl")),
    "--output-prefix", "/raid/s3/opengptx/alexw/Megatron-LM_data/out/oscar",
    "--vocab-file", str(Path(__file__).absolute().parent.parent.joinpath("resources", "tokenization/gpt2-vocab.json")),
    "--dataset-impl", "mmap",
    "--tokenizer-type", "GPT2BPETokenizer",
    "--merge-file", str(Path(__file__).absolute().parent.parent.joinpath("resources", "tokenization/gpt2-merges.txt")),
    "--append-eod",
    "--workers", "8"
]
])
def test_tokenization(arg_values, mocker):
    mocker.patch("sys.argv", [Path(__file__).name] + arg_values)
    main()