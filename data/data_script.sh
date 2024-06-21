#!/bin/bash


echo "Run data generation"

#gsm8k
python all_in_one.py --batch_id=0 --device=0 --repeat 3 --add_uncorrupted_llama --fixed_introducer &
python all_in_one.py --batch_id=1 --device=1 --repeat 3 --add_uncorrupted_llama --fixed_introducer &
python all_in_one.py --batch_id=2 --device=2 --repeat 3 --add_uncorrupted_llama --fixed_introducer &
python all_in_one.py --batch_id=3 --device=3 --repeat 3 --add_uncorrupted_llama --fixed_introducer &
python all_in_one.py --batch_id=4 --device=4 --repeat 3 --add_uncorrupted_llama --fixed_introducer &
python all_in_one.py --batch_id=5 --device=5 --repeat 3 --add_uncorrupted_llama --fixed_introducer &
python all_in_one.py --batch_id=6 --device=6 --repeat 3 --add_uncorrupted_llama --fixed_introducer &
python all_in_one.py --batch_id=7 --device=7 --repeat 3 --add_uncorrupted_llama --fixed_introducer &

#auqa
#python all_in_one.py --batch_id=0 --device=0 --repeat 3 --add_uncorrupted_llama --dataset aqua &
#python all_in_one.py --batch_id=1 --device=1 --repeat 3 --add_uncorrupted_llama --dataset aqua &
#python all_in_one.py --batch_id=2 --device=2 --repeat 3 --add_uncorrupted_llama --dataset aqua &
#python all_in_one.py --batch_id=3 --device=3 --repeat 3 --add_uncorrupted_llama --dataset aqua &
#python all_in_one.py --batch_id=4 --device=4 --repeat 3 --add_uncorrupted_llama --dataset aqua &
#python all_in_one.py --batch_id=5 --device=5 --repeat 3 --add_uncorrupted_llama --dataset aqua &
#python all_in_one.py --batch_id=6 --device=6 --repeat 3 --add_uncorrupted_llama --dataset aqua &
#python all_in_one.py --batch_id=7 --device=7 --repeat 3 --add_uncorrupted_llama --dataset aqua &