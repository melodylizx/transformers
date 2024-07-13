#!/bin/bash
#SBATCH --account=rrg-tyrell-ab
#SBATCH --mem=60G       
#SBATCH --gres=gpu:a100:4              
######SBATCH --time=48:00:00
#SBATCH --time=120:00:00
#SBATCH --cpus-per-gpu=4
#SBATCH -o /scratch/melodyli/fineweb-slurm-%j.out

module load python/3.11.5
module load cuda/12.2
module load arrow
source llm/bin/activate
cd projects/def-tyrell/melodyli/transformers/examples/pytorch/language-modeling/
export HF_DATASETS_CACHE=$SCRATCH/llms/hf_cache

torchrun --nproc_per_node=4 run_clm.py \
    --dataset_name HuggingFaceFW/fineweb \
    --dataset_config_name sample-10BT \
    --per_device_train_batch_size 12 \
    --per_device_eval_batch_size 12 \
    --ddp_timeout 18000 \
    --do_train \
    --do_eval \
    --output_dir /home/melodyli/scratch/fineweb \
    --model_type gpt2 \
    --tokenizer_name tokenizer.json \
    --num_train_epochs 30 \
    --fp16 \
    --learning_rate 6e-4 \
    --lr_scheduler_type cosine \
    --warmup_steps 715 \
    --save_steps 5000  \
    --eval_steps 5000
