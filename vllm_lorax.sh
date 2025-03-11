CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port 8089 \
    --model ./models/Qwen2-7B-Instruct/ \
    --served-model-name Qwen2_7B \
    --enable-lora \
    --lora-modules intent=llm_finetune/lora/CLASSIFY_LORA/saves/ckpt-7b/intent-v2/checkpoint-660\
    keywords=llm_finetune/lora/KEYWORDS_LORA/saves/ckpt-7b/keywords-v2/checkpoint-2500\
    nl2sql=llm_finetune/lora/NL2SQL_LORA/saves/ckpt-7b/no2sql-v2/checkpoint-250\
    --gpu-memory-utilization 0.7 \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --max-model-len 2560 \
    --dtype=half
