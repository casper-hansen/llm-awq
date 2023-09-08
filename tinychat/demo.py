import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, modeling_utils
from attributedict.collections import AttributeDict
from tinychat.stream_generators import StreamGenerator, FalconStreamGenerator
import tinychat.utils.constants
from tinychat.utils.load_quant import load_awq_model, load_awq_llama_fast
from tinychat.utils.prompt_templates import get_prompter, get_stop_token_ids
from tinychat.utils.tune import device_warmup, tune_all_wqlinears

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# opt_params in TinyLLMEngine
gen_params = AttributeDict(
    [
        ("seed", -1),  # RNG seed
        ("n_threads", 1),  # TODO: fix this
        ("n_predict", 512),  # new tokens to predict
        ("n_parts", -1),  # amount of model parts (-1: determine from model dimensions)
        ("n_ctx", 512),  # context size
        ("n_batch", 512),  # batch size for prompt processing (must be >=32 to use BLAS)
        ("n_keep", 0),  # number of tokens to keep from initial prompt
        ("n_vocab", 50272),  # vocabulary size
        # sampling parameters
        ("logit_bias", dict()),  # logit bias for specific tokens: <int, float>
        ("top_k", 40),  # <= 0 to use vocab size
        ("top_p", 0.95),  # 1.0 = disabled
        ("tfs_z", 1.00),  # 1.0 = disabled
        ("typical_p", 1.00),  # 1.0 = disabled
        ("temp", 0.70),  # 1.0 = disabled
        ("repeat_penalty", 1.10),  # 1.0 = disabled
        (
            "repeat_last_n",
            64,
        ),  # last n tokens to penalize (0 = disable penalty, -1 = context size)
        ("frequency_penalty", 0.00),  # 0.0 = disabled
        ("presence_penalty", 0.00),  # 0.0 = disabled
        ("mirostat", 0),  # 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
        ("mirostat_tau", 5.00),  # target entropy
        ("mirostat_eta", 0.10),  # learning rate
    ]
)


def stream_output(output_stream):
    print(f"ASSISTANT: ", end="", flush=True)
    pre = 0
    for outputs in output_stream:
        output_text = outputs["text"]
        output_text = output_text.strip().split(" ")
        now = len(output_text) - 1
        if now > pre:
            print(" ".join(output_text[pre:now]), end=" ", flush=True)
            pre = now
    print(" ".join(output_text[pre:]), flush=True)
    if "timing" in outputs and outputs["timing"] is not None:
        timing = outputs["timing"]
        context_tokens = timing["context_tokens"]
        context_time = timing["context_time"]
        total_tokens = timing["total_tokens"]
        generation_time_list = timing["generation_time_list"]
        generation_tokens = len(generation_time_list)
        average_speed = (context_time + np.sum(generation_time_list)) / (
            context_tokens + generation_tokens
        )
        print("=" * 50)
        print("Speed of Inference")
        print("-" * 50)
        print(f"Context Stage    : {context_time/context_tokens * 1000:.2f} ms/token, {1000/(context_time/context_tokens * 1000):.2f} tokens/s")
        print(
            f"Generation Stage : {np.average(generation_time_list) * 1000:.2f} ms/token, {1000/(np.average(generation_time_list) * 1000):.2f} tokens/s"
        )
        # print(f"Average Speed    : {average_speed * 1000:.2f} ms/token")
        print("=" * 50)
        # print("token num:", total_tokens)
        # print("Model total Time = ", (context_time + np.sum(generation_time_list))*1000, "ms" )
    return " ".join(output_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type", type=str, default="LLaMa", help="type of the model"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/data/llm/checkpoints/vicuna-hf/vicuna-7b",
        help="path to the model",
    )
    parser.add_argument(
        "--precision", type=str, default="W4A16", help="compute precision"
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--q_group_size", type=int, default=128)
    parser.add_argument(
        "--load_quant",
        type=str,
        default="/data/llm/checkpoints/vicuna-hf/vicuna-7b-awq-w4g128.pt",
        help="path to the pre-quanted 4-bit weights",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=2048,
        help="maximum sequence length for kv cache"
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=1,
        help="maximum batch size for kv cache"
    )

    args = parser.parse_args()
    assert args.model_type.lower() in [
        "llama",
        "falcon",
        "mpt",
    ], "We only support llama & falcon & mpt now"
    assert args.precision in ["W4A16", "W16A16"], "We only support W4A16/W16A16 now"

    gen_params.n_predict = 512
    gen_params.n_vocab = 32000
    tinychat.utils.constants.max_batch_size = args.max_batch_size
    tinychat.utils.constants.max_seq_len = args.max_seq_len

    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.kaiming_normal_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, use_fast=False, trust_remote_code=True
        )
    modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    model = load_awq_llama_fast(
        model, args.load_quant, 4, args.q_group_size, args.device
    )

    # device warm up
    device_warmup(args.device)

    stream_generator = StreamGenerator

    # Optimize AWQ quantized model
    if args.precision == "W4A16" and args.model_type.lower() == "llama":
        from tinychat.modules import make_quant_norm, make_quant_attn, make_fused_mlp

        make_quant_attn(model, args.device)
        make_quant_norm(model)
        make_fused_mlp(model)
    
    @torch.inference_mode()
    def new():
        def _timer(func):
            start = time.time()
            out = func()
            return out, time.time() - start

        def _generate(model, model_out, n_generate, batch_size):
            past_key_values = model_out.past_key_values

            for i in range(n_generate):
                logits = model_out.logits[:, -1, :]
                new_tokens = []

                for batch_index in range(batch_size):
                    probs = torch.softmax(logits[batch_index], dim=-1)
                    token = torch.multinomial(probs, num_samples=1)
                    new_tokens.append(token)
                
                tokens = torch.as_tensor(new_tokens, device=args.device).unsqueeze(-1)

                model_out = model(tokens, use_cache=True, past_key_values=past_key_values)

        def _warmup(device:str):
            warm_up = torch.randn((4096,4096)).to(device)
            torch.mm(warm_up,warm_up)
        
        n_generate = 128
        n_context = 256
        batch_size = 1

        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        _warmup(args.device)

        # Generate random inputs
        n_context = n_context - n_generate
        ids = torch.randint(0, tokenizer.vocab_size, (batch_size, n_context)).cuda()

        # Context stage
        model_out, context_time = _timer(lambda: model(ids, use_cache=True))

        # Generation stage
        _, generation_time = _timer(lambda: _generate(model, model_out, n_generate, batch_size))

        # Prints
        memory_used = torch.cuda.max_memory_allocated(args.device) / (1024 ** 2)
        context_tokens_per_second = n_context / context_time * batch_size
        context_ms_per_token = (context_time*1000) / n_context / batch_size
        inference_tokens_per_second = n_generate / generation_time * batch_size
        inference_ms_per_token = (generation_time*1000) / n_generate / batch_size

        print(f"[======] Model summary: {args.model_path} [======]")
        print(f"[*] Context speed: {context_tokens_per_second:.2f} tokens/second ({context_ms_per_token:.2f} ms/token)")
        print(f"[*] Generation speed: {inference_tokens_per_second:.2f} tokens/second ({inference_ms_per_token:.2f} ms/token)")
        print(f"[*] VRAM: {memory_used:.2f} MB")

    def old():
        model_prompter = get_prompter(args.model_type, args.model_path)
        stop_token_ids = get_stop_token_ids(args.model_type, args.model_path)
        count = 0
        while True:
            # Get input from the user
            input_prompt = "Tell me about the flying blue wolf"
            model_prompter.insert_prompt(input_prompt)
            output_stream = stream_generator(
                model,
                tokenizer,
                model_prompter.model_input,
                gen_params,
                device=args.device,
                stop_token_ids=stop_token_ids,
            )
            outputs = stream_output(output_stream)
            model_prompter.update_template(outputs)
            count += 1
    
    new()