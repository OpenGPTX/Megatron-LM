# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
import datetime
import threading
import traceback

import torch
from flask import Flask, jsonify, request
from flask_restful import Api, Resource

from megatron import get_args, get_tokenizer
from megatron.text_generation import (
    beam_search_and_post_process,
    generate_and_post_process,
)

GENERATE_NUM = 0
BEAM_NUM = 1
lock = threading.Lock()


class MegatronGenerate(Resource):
    def __init__(self, model):
        self.model = model

    @staticmethod
    def send_do_generate():
        choice = torch.cuda.LongTensor([GENERATE_NUM])
        torch.distributed.broadcast(choice, 0)

    @staticmethod
    def send_do_beam_search():
        choice = torch.cuda.LongTensor([BEAM_NUM])
        torch.distributed.broadcast(choice, 0)

    def put(self):
        args = get_args()

        if "prompts" not in request.get_json():
            return "prompts argument required", 400

        if "max_len" in request.get_json():
            return "max_len is no longer used.  Replace with tokens_to_generate", 400

        if "sentences" in request.get_json():
            return "sentences is no longer used.  Replace with prompts", 400

        prompts = request.get_json()["prompts"]
        if not isinstance(prompts, list):
            return "prompts is not a list of strings", 400

        if len(prompts) == 0:
            return "prompts is empty", 400

        if len(prompts) > 128:
            return "Maximum number of prompts is 128", 400

        tokens_to_generate = (
            64  # Choosing hopefully sane default.  Full sequence is slow
        )

        if "tokens_to_generate" in request.get_json():
            tokens_to_generate = request.get_json()["tokens_to_generate"]
            if not isinstance(tokens_to_generate, int):
                return "tokens_to_generate must be an integer greater than 0", 400
            if tokens_to_generate < 0:
                return (
                    "tokens_to_generate must be an integer greater than or equal to 0",
                    400,
                )

        logprobs = False
        if "logprobs" in request.get_json():
            logprobs = request.get_json()["logprobs"]
            if not isinstance(logprobs, bool):
                return "logprobs must be a boolean value", 400

        if tokens_to_generate == 0 and not logprobs:
            return "tokens_to_generate=0 implies logprobs should be True", 400

        temperature = 1.0
        if "temperature" in request.get_json():
            temperature = request.get_json()["temperature"]
            if not (type(temperature) == int or type(temperature) == float):
                return (
                    "temperature must be a positive number less than or equal to 100.0",
                    400,
                )
            if not (0.0 < temperature <= 100.0):
                return (
                    "temperature must be a positive number less than or equal to 100.0",
                    400,
                )

        top_k = 0.0
        if "top_k" in request.get_json():
            top_k = request.get_json()["top_k"]
            if not (type(top_k) == int):
                return (
                    "top_k must be an integer equal to or greater than 0 and less than or equal to 1000",
                    400,
                )
            if not (0 <= top_k <= 1000):
                return (
                    "top_k must be equal to or greater than 0 and less than or equal to 1000",
                    400,
                )

        top_p = 0.0
        if "top_p" in request.get_json():
            top_p = request.get_json()["top_p"]
            if not (type(top_p) == float):
                return "top_p must be a positive float less than or equal to 1.0", 400
            if top_p > 0.0 and top_k > 0.0:
                return "cannot set both top-k and top-p samplings.", 400
            if not (0 <= top_p <= 1.0):
                return "top_p must be less than or equal to 1.0", 400

        top_p_decay = 0.0
        if "top_p_decay" in request.get_json():
            top_p_decay = request.get_json()["top_p_decay"]
            if not (type(top_p_decay) == float):
                return (
                    "top_p_decay must be a positive float less than or equal to 1.0",
                    400,
                )
            if top_p == 0.0:
                return "top_p_decay cannot be set without top_p", 400
            if not (0 <= top_p_decay <= 1.0):
                return "top_p_decay must be less than or equal to 1.0", 400

        top_p_bound = 0.0
        if "top_p_bound" in request.get_json():
            top_p_bound = request.get_json()["top_p_bound"]
            if not (type(top_p_bound) == float):
                return (
                    "top_p_bound must be a positive float less than or equal to top_p",
                    400,
                )
            if top_p == 0.0:
                return "top_p_bound cannot be set without top_p", 400
            if not (0.0 < top_p_bound <= top_p):
                return "top_p_bound must be greater than 0 and less than top_p", 400

        add_BOS = False
        if "add_BOS" in request.get_json():
            add_BOS = request.get_json()["add_BOS"]
            if not isinstance(add_BOS, bool):
                return "add_BOS must be a boolean value", 400

        if any([len(prompt) == 0 for prompt in prompts]) and not add_BOS:
            return "Empty prompts require add_BOS=true"

        stop_on_double_eol = False
        if "stop_on_double_eol" in request.get_json():
            stop_on_double_eol = request.get_json()["stop_on_double_eol"]
            if not isinstance(stop_on_double_eol, bool):
                return "stop_on_double_eol must be a boolean value", 400

        stop_on_eol = False
        if "stop_on_eol" in request.get_json():
            stop_on_eol = request.get_json()["stop_on_eol"]
            if not isinstance(stop_on_eol, bool):
                return "stop_on_eol must be a boolean value", 400

        prevent_newline_after_colon = False
        if "prevent_newline_after_colon" in request.get_json():
            prevent_newline_after_colon = request.get_json()[
                "prevent_newline_after_colon"
            ]
            if not isinstance(prevent_newline_after_colon, bool):
                return "prevent_newline_after_colon must be a boolean value", 400

        random_seed = -1
        if "random_seed" in request.get_json():
            random_seed = request.get_json()["random_seed"]
            if not isinstance(random_seed, int):
                return "random_seed must be integer", 400
            if random_seed < 0:
                return "random_seed must be a positive integer", 400

        no_log = False
        if "no_log" in request.get_json():
            no_log = request.get_json()["no_log"]
            if not isinstance(no_log, bool):
                return "no_log must be a boolean value", 400

        beam_width = None
        if "beam_width" in request.get_json():
            beam_width = request.get_json()["beam_width"]
            if not isinstance(beam_width, int):
                return "beam_width must be integer", 400
            if beam_width < 1:
                return "beam_width must be an integer > 1", 400
            if len(prompts) > 1:
                return "When doing beam_search, batch size must be 1", 400

        stop_token = get_tokenizer().eod
        if "stop_token" in request.get_json():
            stop_token = request.get_json()["stop_token"]
            if not isinstance(stop_token, int):
                return "stop_token must be an integer", 400

        length_penalty = 1
        if "length_penalty" in request.get_json():
            length_penalty = request.get_json()["length_penalty"]
            if not isinstance(length_penalty, float):
                return "length_penalty must be a float", 400

        with lock:  # Need to get lock to keep multiple threads from hitting code
            if not no_log:
                print("request IP: " + str(request.remote_addr))
                # print(json.dumps(request.get_json()), flush=True)
                print("start time: ", datetime.datetime.now())

            try:
                if beam_width is not None:
                    MegatronGenerate.send_do_beam_search()  # Tell other ranks we're doing beam_search
                    (
                        response,
                        response_seg,
                        response_scores,
                        response_tokens,
                    ) = beam_search_and_post_process(
                        self.model,
                        prompts=prompts,
                        tokens_to_generate=tokens_to_generate,
                        beam_size=beam_width,
                        add_BOS=add_BOS,
                        stop_token=stop_token,
                        num_return_gen=beam_width,  # Returning whole beam
                        length_penalty=length_penalty,
                        prevent_newline_after_colon=prevent_newline_after_colon,
                    )

                    result = {
                        "text": response,
                        "segments": response_seg,
                        "scores": response_scores,
                        "tokens": response_tokens,
                    }

                else:
                    MegatronGenerate.send_do_generate()  # Tell other ranks we're doing generate

                    return_is_max_logprobs = tokens_to_generate == 0

                    generate_result = generate_and_post_process(
                        self.model,
                        prompts=prompts,
                        tokens_to_generate=tokens_to_generate,
                        return_output_log_probs=logprobs,
                        top_k_sampling=top_k,
                        top_p_sampling=top_p,
                        top_p_decay=top_p_decay,
                        top_p_bound=top_p_bound,
                        temperature=temperature,
                        add_BOS=add_BOS,
                        use_eod_token_for_early_termination=True,
                        stop_on_double_eol=stop_on_double_eol,
                        stop_on_eol=stop_on_eol,
                        prevent_newline_after_colon=prevent_newline_after_colon,
                        random_seed=random_seed,
                        return_is_max_logprobs=return_is_max_logprobs,
                    )

                    if return_is_max_logprobs:
                        (
                            response,
                            response_seg,
                            response_logprobs,
                            response_tokens,
                            response_is_max_logprobs,
                        ) = generate_result
                    else:
                        (
                            response,
                            response_seg,
                            response_logprobs,
                            response_tokens,
                        ) = generate_result
                        response_is_max_logprobs = None

                    result = {
                        "text": response,
                        "segments": response_seg,
                        "logprobs": response_logprobs,
                        "tokens": response_tokens,
                        "is_max_logprobs": response_is_max_logprobs,
                    }
                return jsonify(result)

            except ValueError as ve:
                traceback.print_exc()
                return ve.args[0]
            print("end time: ", datetime.datetime.now())


class MegatronTokenizer(Resource):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def put(self):
        if "prompts" not in request.get_json():
            return "prompts argument required", 400

        prompts = request.get_json()["prompts"]
        if not isinstance(prompts, list):
            return "prompts is not a list of strings", 400

        if len(prompts) == 0:
            return "prompts is empty", 400

        return jsonify(
            {"tokens": [self.tokenizer.tokenize(prompt) for prompt in prompts]}
        )


class MegatronDetokenizer(Resource):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def put(self):
        if "seqs" not in request.get_json():
            return "seqs argument required", 400

        seqs = request.get_json()["seqs"]
        if not isinstance(seqs, list):
            return "seqs is not a list of strings", 400

        if len(seqs) == 0:
            return "seqs is empty", 400

        return jsonify({"text": [self.tokenizer.detokenize(seq) for seq in seqs]})

    def options():
        return jsonify({})


class MegatronMetadata(Resource):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def put(self):
        args = get_args()

        try:
            if hasattr(args, "eos_id"):
                eod_id = args.eos_id
            else:
                try:
                    eod_id = self.tokenizer.eod
                except NotImplementedError:
                    eod_id = self.tokenizer.eos_token_id
            max_length = args.max_position_embeddings
            if max_length is None:
                max_length = args.seq_length

            return jsonify(
                {
                    "vocab_size": self.tokenizer.vocab_size,
                    "eod_token_id": eod_id,
                    "max_length": max_length,
                }
            )
        except ValueError as ve:
            return ve.args[0], 300


class MegatronServer(object):
    def __init__(self, model):
        self.app = Flask(__name__, static_url_path="")
        api = Api(self.app)
        api.add_resource(MegatronGenerate, "/api", resource_class_args=[model])
        api.add_resource(
            MegatronTokenizer, "/tokenize", resource_class_args=[get_tokenizer()]
        )
        api.add_resource(
            MegatronDetokenizer, "/detokenize", resource_class_args=[get_tokenizer()]
        )
        api.add_resource(
            MegatronMetadata, "/metadata", resource_class_args=[get_tokenizer()]
        )

    def run(self, url):
        self.app.run(url, threaded=True, debug=False)
