#!/bin/bash
#SBATCH --account=rrg-tyrell-ab
#SBATCH --mem=60G       
#SBATCH --gres=gpu:a100:4              
######SBATCH --time=48:00:00
#SBATCH --time=120:00:00
#SBATCH --cpus-per-gpu=4
#SBATCH -o /scratch/melodyli/wiki-slurm-%j.out


source llm/bin/activate
module load arrow
cd projects/def-tyrell/melodyli/transformers/examples/pytorch/language-modeling/


torchrun --nproc_per_node=4 run_clm.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 24 \
    --per_device_eval_batch_size 24 \
    --ddp_timeout 18000 \
    --do_train \
    --do_eval \
    --output_dir /home/melodyli/scratch/gpt2_ckpts_20000 \
    --model_type gpt2 \
    --tokenizer_name tokenizer.json \
    --num_train_epochs 5 \
    --fp16 \
    --learning_rate 6e-4 \
    --lr_scheduler_type cosine \
    --warmup_steps 715 \
    --save_steps 5000  \
    --eval_steps 5000

