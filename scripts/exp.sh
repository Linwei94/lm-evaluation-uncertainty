cd ..
lm-eval --model hf --model_args pretrained=GSAI-ML/LLaDA-1.5,trust_remote_code=True --tasks mmlu --device cuda:0 --batch_size 8 --output_path results/mmlu
lm-eval --model hf --model_args pretrained=Dream-org/Dream-v0-Base-7B,trust_remote_code=True --tasks mmlu --device cuda:0 --batch_size 8 --output_path results/mmlu
lm-eval --model hf --model_args pretrained=Dream-org/Dream-v0-Instruct-7B,trust_remote_code=True --tasks mmlu --device cuda:0 --batch_size 8 --output_path results/mmlu
lm-eval --model hf --model_args pretrained=Qwen/Qwen2.5-7B-Instruct,trust_remote_code=True --tasks mmlu --device cuda:0 --batch_size 8 --output_path results/mmlu
lm-eval --model hf --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct,trust_remote_code=True --tasks mmlu --device cuda:0 --batch_size 8 --output_path results/mmlu
lm-eval --model hf --model_args pretrained=meta-llama/Llama-3.1-8B,trust_remote_code=True --tasks mmlu --device cuda:0 --batch_size 8 --output_path results/mmlu

# mmlu vnc
lm-eval --model hf   --model_args pretrained=Qwen/Qwen2.5-7B-Instruct,trust_remote_code=True   --tasks mmlu_vnc   --device cuda:0   --batch_size 8   --output_path results/mmlu_vnc
lm-eval --model hf   --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct,trust_remote_code=True   --tasks mmlu_vnc   --device cuda:1   --batch_size 8   --output_path results/mmlu_vnc