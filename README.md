# LLM-Neo


This repo contains the code for our paper: <a href="https://arxiv.org/abs/2411.06839" target="_blank">LLM-Neo: Parameter Efficient Knowledge Distillation for Large Language Models</a> by <a href="https://wutaiqiang.github.io" target="_blank">Runming Yang</a>, Taiqiang Wu, Jiahao Wang, Pengfei Hu, Yik Chung Wu, Ngai Wong and Yujiu Yang.

There is an <a href="https://zhuanlan.zhihu.com/p/8642629256" target="_blank"> explaination blog </a> for this paper (in Chinese).


## Overview




Thanks for solid and easy training framework by LLaMA-Factory! Our code is basiclly build on them.


Environment:


Quick start:
```
cd LLaMA-Factory
python3 script_Neo.py
bash run_train.sh
```
The script_Neo.py can generate for all 4 training-strategy once time, the yaml will be generated to examples/train_neo and you can modify them manually.

Follow the guideline

Advanced usage:
```
# for basic hypermeters
python3 script_Neo.py --base_lr 1e-5 --epochs 3 --batch_size 16 --grad_accum 4 --max_samples 1000 

# for LoRA and Neo
python3 script_Neo.py --run lora neo --lora_rank 32 --base_model meta-llama/Meta-Llama-3-8B-Instruct

# for KD and Neo
python3 script_Neo.py --run kd neo --base_model meta-llama/Meta-Llama-3-8B-Instruct --teacher_model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Llama-8B 
```

LLM-Neo is the combination of LoRA and KD, which is not originally supported by LLaMA-Factory. 
We add the teacher_model and kd_ratio parameters in src/llamafactory/hparams/finetuning_args and src/llamafactory/train/sft/trainer, which is easy to extend to other methods when you DIY.

