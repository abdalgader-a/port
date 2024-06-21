#!/bin/bash

echo "Run *O Experiments (PEFT)"

#preference
accelerate launch dpo.py --scheme=corr_only_1 --model_name_or_path=Abdalgader/sft_falcon-11B --per_device_train_batch_size 2 --learning_rate 8e-6 --gradient_accumulation_steps 2 --logging_steps 10 --eval_steps 500 --output_dir=corrupted_corr_only_1_falcon11b_lora_beta0.2_1epoch --optim=rmsprop --warmup_steps=10 --bf16 --logging_first_step --no_remove_unused_columns --use_peft --lora_r=64 --lora_alpha=16 --max_prompt_length=1024 --max_length=2048 --report_to=wandb --num_train_epochs 1 --beta 0.2

#kto
accelerate launch kto.py --scheme=kto_corr_only_1 --model_name_or_path=Abdalgader/sft_falcon-11B --per_device_train_batch_size 2 --learning_rate 8e-6 --gradient_accumulation_steps 2 --logging_steps 10 --eval_steps 500 --output_dir=kto_corr_only_1_falcon_11B_gsm8k_beta0.5_1epochs --optim=rmsprop --warmup_steps=10 --bf16 --logging_first_step --no_remove_unused_columns --use_peft --lora_r=64 --lora_alpha=16 --max_prompt_length=1024 --max_length=2048 --report_to=wandb --num_train_epochs 1 --beta 0.5

#orpo
accelerate launch orpo.py --scheme=corr_only_1 --model_name_or_path=tiiuae/falcon-11B --per_device_train_batch_size 2 --learning_rate 1e-8 --gradient_accumulation_steps 2 --logging_steps 10 --eval_steps 500 --output_dir=orpo_beta0.001_lr1e8_1epochs_corr_only_1_falcon_11B_gsm8k --optim=rmsprop --warmup_steps=10 --bf16 --logging_first_step --no_remove_unused_columns --use_peft --lora_r=64 --lora_alpha=16 --max_prompt_length=1024 --max_length=2048 --report_to=wandb --num_train_epochs 1 --beta 0.001

#ipo
accelerate launch dpo.py --loss_type ipo --scheme=corr_only_1 --model_name_or_path=Abdalgader/sft_falcon-11B --per_device_train_batch_size 2 --learning_rate 8e-6 --gradient_accumulation_steps 2 --logging_steps 10 --eval_steps 500 --output_dir=ipo_beta0.7_corr_only_1_falcon_11B_gsm8k_1epochs --optim=rmsprop --warmup_steps=10 --bf16 --logging_first_step --no_remove_unused_columns --use_peft --lora_r=64 --lora_alpha=16 --max_prompt_length=1024 --max_length=2048 --report_to=wandb --num_train_epochs 1 --beta 0.7