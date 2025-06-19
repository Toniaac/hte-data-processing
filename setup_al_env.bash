#!/bin/bash


#%%%%%%%%%%%%%% set up envivronment %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
env_name="llm-al"
project="$HOME/projects/def-j3goals/$USER"
env_path="$project/.conda/envs/$env_name"
pip_cache_dir="$project/.cache/pip"
alias pip="pip --cache-dir $pip_cache_dir"

module load python/3.9  scipy-stack
mkdir -p $env_path
virtualenv --no-download $env_path
source $env_path/bin/activate 
export HF_HOME="$project/.cache/huggingface"

pip install ipython transformers torch einops accelerate Xformers triton scikit-learn xgboost json glob re -no-index

