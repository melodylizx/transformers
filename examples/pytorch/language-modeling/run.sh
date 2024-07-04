#!/bin/bash
#SBATCH --account=rrg-tyrell-ab
#SBATCH --mem=32G       
#SBATCH --gres=gpu:2                   # Ask for 1 GPU
######SBATCH --time=48:00:00
#BATCH --time=120:00:00
#SBATCH --cpus-per-gpu=2
#SBATCH --output=sbatch_out/run1.%A.%a.out
#SBATCH --error=sbatch_err/run1.%A.%a.err
#SBATCH --job-name=run1


conda activate llm

torchrun --nproc_per_node=2 run_clm.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 24 \
    --per_device_eval_batch_size 24 \
    --do_train \
    --do_eval \
    --output_dir /network/scratch/z/zixuan.li/gpt2_ckpts_20000 \
    --model_type gpt2 \
    --tokenizer_name tokenizer.json \
    --num_train_epochs 20000 \
    --fp16 \
    --learning_rate 6e-4 \
    --lr_scheduler_type cosine \
    --warmup_steps 715 \
    --save_steps 5000  \
    --eval_steps 5000

