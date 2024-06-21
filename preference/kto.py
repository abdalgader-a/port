# flake8: noqa
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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


import logging
import os
from contextlib import nullcontext

import sys
from pathlib import Path
sys.path.append(str(Path('.').absolute().parent))

from data.extraction import data_extractor, ExtractorScriptArguments

TRL_USE_RICH = os.environ.get("TRL_USE_RICH", False)

from trl.commands.cli_utils import init_zero_verbose, TrlParser

if TRL_USE_RICH:
    init_zero_verbose()
    FORMAT = "%(message)s"

    from rich.console import Console
    from rich.logging import RichHandler

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl import (
    KTOConfig,
    KTOTrainer,
    ModelConfig,
    RichProgressCallback,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

if TRL_USE_RICH:
    logging.basicConfig(format=FORMAT, datefmt="[%X]", handlers=[RichHandler()], level=logging.INFO)

if __name__ == "__main__":
    parser = TrlParser((KTOConfig, ModelConfig, ExtractorScriptArguments))
    training_args, model_config, extractor_args = parser.parse_args_and_config()

    training_args.gradient_checkpointing_kwargs = {'use_reentrant': False}
    training_args.ddp_find_unused_parameters = False

    # Force use our print callback
    if TRL_USE_RICH:
        training_args.disable_tqdm = True
        console = Console()

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    peft_config = get_peft_config(model_config)
    if peft_config is None:
        model_ref = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    else:
        model_ref = None
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% for message in messages %}{{message['role'] + ': ' + message['content'] + '\n\n'}}{% endfor %}{{ eos_token }}"

    model_path = model.base_model.name_or_path
    if "/" in model_path:
        model_name = model_path[model_path.rindex('/') + 1:]
    else:
        model_name = model_path

    os.environ["WANDB_PROJECT"] = model_name + "-KTO-reasoning"

    ################
    # Optional rich context managers
    ###############
    init_context = nullcontext() if not TRL_USE_RICH else console.status("[bold green]Initializing the KTOTrainer...")
    save_context = (
        nullcontext()
        if not TRL_USE_RICH
        else console.status(f"[bold green]Training completed! Saving the model to {training_args.output_dir}")
    )

    ################
    # Dataset
    ################
    train_data = Dataset.from_pandas(
        data_extractor(tokenizer, scheme=extractor_args.scheme, model=extractor_args.llm, split='train', dataset=extractor_args.dataset),
        preserve_index=False)
    eval_data = Dataset.from_pandas(
        data_extractor(tokenizer, scheme=extractor_args.scheme, model=extractor_args.llm, split='test', dataset=extractor_args.dataset),
        preserve_index=False)

    train_data.shuffle(seed=42)
    eval_data.shuffle(seed=42)

    ################
    # Training
    ################
    with init_context:
        print("Initiate trainer...")
        trainer = KTOTrainer(
            model,
            model_ref,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=eval_data,
            tokenizer=tokenizer,
            peft_config=get_peft_config(model_config),
            callbacks=[RichProgressCallback] if TRL_USE_RICH else None,
        )
    print("Strating training...")
    trainer.train()

    with save_context:
        trainer.save_model(training_args.output_dir)
