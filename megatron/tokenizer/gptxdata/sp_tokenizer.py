from __future__ import annotations
import os

from pathlib import Path
from typing import Dict, List, Mapping, Optional, Union

from datasets import load_dataset
import numpy as np
import sentencepiece as spm
import sentencepiece.sentencepiece_model_pb2 as model
import torch

from .gptx_tokenizer import GPTXTokenizer


def stream_from_json_dataset(datasets_dir: str, key: str, cache_dir: str):
    for d in load_dataset(path=datasets_dir, split="train", cache_dir=cache_dir):
        yield d[key]


class SPTokenizer(GPTXTokenizer):
    _vocab: Mapping[str, int] = None

    def __init__(self, tokenizer: Optional[spm.SentencePieceProcessor] = None):
        self.tokenizer = tokenizer
        self.tokenizer_config = None
        self.continuation_tokenizer = None
        self.add_prefix_space = None
        self.read_model_proto()

    def read_model_proto(self):
        if self.tokenizer is not None:
            proto = model.ModelProto()
            proto.ParseFromString(self.tokenizer.serialized_model_proto())
            # Variant without additional sp processor
            self.add_prefix_space = proto.normalizer_spec.add_dummy_prefix

            # Variant with additional sp processor
            proto.normalizer_spec.add_dummy_prefix = False

            self.continuation_tokenizer = spm.SentencePieceProcessor()
            self.continuation_tokenizer.LoadFromSerializedProto(
                proto.SerializeToString()
            )

    def train(
        self,
        datasets_dir: str,
        save_dir: Path,
        tokenizer_config: Path,
        text_key: str,
        cache_dir: str = "./",
    ):
        # read config and create save_dir
        self.tokenizer_config = self.load_json(tokenizer_config)
        self.add_special_tokens_to_config(config=self.tokenizer_config)
        self.create_dir(save_dir)

        print("Train tokenizer")
        model_prefix = save_dir.joinpath(
            f'{self.tokenizer_config["model_type"]}_tokenizer'
        )
        spm.SentencePieceTrainer.train(
            sentence_iterator=stream_from_json_dataset(
                datasets_dir=datasets_dir, key=text_key, cache_dir=cache_dir
            ),
            model_prefix=model_prefix,
            **self.tokenizer_config,
        )

        # add parameters to config
        self.tokenizer_config["datasets_dir"] = datasets_dir
        self.tokenizer_config["save_dir"] = save_dir
        self.tokenizer_config["text_key"] = text_key
        self.tokenizer_config["cache_dir"] = cache_dir
        self.tokenizer_config["library"] = "sentencepiece"

        print("Save tokenizer")
        self.save_tokenizer_config(save_dir)

        # load tokenizer into processor
        model_file = str(model_prefix) + ".model"
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(model_file)
        self.read_model_proto()
        print("Done")

    def encode(
        self, text: str, return_tokens: bool = False, is_continuation: bool = False
    ):
        assert self.tokenizer is not None, "No tokenizer is currently loaded"

        # Variant without additional sp processor:
        # if is_continuation and self.add_prefix_space:
        #    if text.startswith(" "):
        #        text = text[1:]
        # if return_tokens:
        #     return self.tokenizer.encode_as_pieces(text)
        # else:
        #     return self.tokenizer.encode(text)

        # Variant with additional sp processor:
        tokenizer = self.continuation_tokenizer if is_continuation else self.tokenizer

        if return_tokens:
            return tokenizer.encode_as_pieces(text)
        else:
            return tokenizer.encode(text)

    def decode(
        self,
        token_ids: Union[List[int], List[List[int]]],
        num_threads: Optional[int] = None,
    ) -> str:
        return self.tokenizer.decode(input=token_ids, num_threads=num_threads)

    def batch_decode(
        self,
        token_ids: Union[List[List[int]], torch.Tensor, np.ndarray],
        num_threads: Optional[int] = None,
    ) -> List[str]:
        return self.decode(token_ids=token_ids, num_threads=num_threads)

    def load(self, model_file: str):
        # Load tokenizer into processor
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(model_file)

    @classmethod
    def instantiate_from_file_or_name(cls, model_file_or_name: str):
        return cls(tokenizer=spm.SentencePieceProcessor(model_file=model_file_or_name))

    def save_tokenizer(self, save_dir: str) -> None:
        if not os.path.isdir(save_dir):
            print(f"Vocabulary path ({save_dir}) should be a directory")
            return
        out_vocab_file = os.path.join(save_dir, "tokenizer.model")

        # if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
        #     copyfile(self.vocab_file, out_vocab_file)
        # elif not os.path.isfile(self.vocab_file):
        with open(out_vocab_file, "wb") as f:
            content_spiece_model = self.tokenizer.serialized_model_proto()
            f.write(content_spiece_model)

        return (out_vocab_file,)

    def remove_tokens(self, tokens: List[str]) -> SPTokenizer:
        # TODO
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size()

    @property
    def vocab(self) -> Mapping[str, int]:
        if self._vocab is None:
            self._vocab = {
                self.tokenizer.IdToPiece(i): i for i in range(self.vocab_size)
            }
        return self._vocab

    @property
    def eod(self) -> int:
        return self.tokenizer.PieceToId(self.eod_token)

    def create_list_of_special_tokens(self) -> List[str]:
        return [self.bos_token, self.eos_token, self.pad_token, self.eod_token] + [
            f"<placeholder_tok_{i}>" for i in range(256)
        ]

    def add_special_tokens_to_config(self, config: Dict[str, str]) -> None:
        config["user_defined_symbols"] = self.create_list_of_special_tokens()

    def add_special_tokens(self, tokens: List[str]) -> None:
        proto = model.ModelProto()
        proto.ParseFromString(self.tokenizer.serialized_model_proto())

        for token in tokens:
            sp_token = model.ModelProto().SentencePiece()
            sp_token.piece = token
            sp_token.score = 0
            proto.pieces.append(sp_token)

        self.tokenizer.LoadFromSerializedProto(proto.SerializeToString())
        # In case the special tokens are not added at the end, the
        # vocabularies should be rebuilt.
        self._vocab = None
        self._inv_vocab = None
