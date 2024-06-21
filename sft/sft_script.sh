#!/bin/bash

echo "Run SFT Experiment (PEFT)"

#sft
accelerate launch sft.py --model_name_or_path=tiiuae/falcon-11B --learning_rate=1.41e-5 --per_device_train_batch_size=16 --gradient_accumulation_steps=16 --output_dir=sft_falcon-11b --logging_steps=1 --num_train_epochs=3 --max_steps=-1 --gradient_checkpointing --use_peft --lora_r=64 --lora_alpha=16 --dataset_text_field=prompt --max_seq_length=2048 --report_to=wandb

