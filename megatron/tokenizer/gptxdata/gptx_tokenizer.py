from __future__ import annotations

import abc
import json
from pathlib import Path
from typing import Dict, List, Mapping, Union

import torch
from numpy.typing import NDArray

from .tokenizer_constants import (
    BOS_TOKEN,
    EOD_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
)


class GPTXTokenizer(abc.ABC):
    # TODO: add EOD token
    _inv_vocab: Mapping[int, str] = None

    def __init__(self):
        self.tokenizer_config = None

    @abc.abstractmethod
    def train(
        self,
        datasets_dir: str,
        save_dir: Path,
        tokenizer_config: Path,
        text_key: str,
        cache_dir: str,
    ):
        pass

    @abc.abstractmethod
    def encode(self, text: str, return_tokens: bool):
        pass

    @abc.abstractmethod
    def decode(self, token_ids: Union[List[int], torch.Tensor, NDArray, int]) -> str:
        pass

    @abc.abstractmethod
    def batch_decode(
        self, token_ids: Union[List[List[int]], torch.Tensor, NDArray]
    ) -> List[str]:
        pass

    @abc.abstractmethod
    def load(cls, model_file: str) -> None:
        pass

    @classmethod
    def instantiate_from_file_or_name(cls, model_file_or_name: str) -> None:
        raise NotImplementedError

    def remove_tokens(self, tokens: List[str]) -> GPTXTokenizer:
        # required for sage
        raise NotImplementedError

    def save_tokenizer(self, save_dir: str) -> None:
        # required for sage
        raise NotImplementedError

    def save(self, save_dir: str) -> None:
        # required for sage
        self.save_tokenizer(save_dir=save_dir)
        if self.tokenizer_config is not None:
            self.save_tokenizer_config(save_dir=Path(save_dir))

    def save_tokenizer_config(self, save_dir: Path) -> None:
        # convert Path to str
        for k in self.tokenizer_config:
            if isinstance(self.tokenizer_config[k], Path):
                self.tokenizer_config[k] = str(self.tokenizer_config[k])

        info_file = save_dir / "tokenizer_config.json"
        with info_file.open("w") as f:
            json.dump(self.tokenizer_config, f, indent=4)

    @staticmethod
    def load_json(path: Path) -> dict:
        with path.open("r") as f:
            return json.load(f)

    @staticmethod
    def create_dir(directory: Path) -> None:
        if not directory.exists():
            directory.mkdir(parents=False)

    @property
    def inv_vocab(self) -> Mapping[int, str]:
        if self._inv_vocab is None:
            self._inv_vocab = {value: key for key, value in self.vocab.items()}
        return self._inv_vocab

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def vocab(self) -> Dict[str, int]:
        # required for sage
        raise NotImplementedError

    @property
    def backend_tokenizer(self) -> Tokenizer:
        raise NotImplementedError

    @property
    def eod(self) -> int:
        pass

    @abc.abstractmethod
    def create_list_of_special_tokens(self) -> List[str]:
        pass

    @abc.abstractmethod
    def add_special_tokens_to_config(self, config: Dict[str, str]):
        pass

    @abc.abstractmethod
    def add_special_tokens(self, tokens: List[str]) -> None:
        ...

    @property
    def eod_token(self) -> str:
        return EOD_TOKEN

    @property
    def bos_token(self) -> str:
        return BOS_TOKEN

    @property
    def eos_token(self) -> str:
        return EOS_TOKEN

    @property
    def pad_token(self) -> str:
        return PAD_TOKEN
