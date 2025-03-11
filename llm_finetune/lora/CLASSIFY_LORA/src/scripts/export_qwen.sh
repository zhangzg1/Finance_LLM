CUDA_VISIBLE_DEVICES=0 python src/export_model.py \
    --model_name_or_path ../../../models/Qwen2-7B-Instruct/ \
    --template qwen \
    --finetuning_type lora \
    --lora_target all \
    --export_dir ./saves/merge_qwen2-7b-intent-v2/ \
    --adapter_name_or_path ./saves/ckpt-7b/intent-v2/checkpoint-660/
