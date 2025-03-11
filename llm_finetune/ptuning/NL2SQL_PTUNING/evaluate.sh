PRE_SEQ_LEN=128
CHECKPOINT=Fin-Train-chatglm3-6b-pt-128-2e-2
CUDA_VISIBLE_DEVICES=0 python main.py \
    --do_predict \
    --validation_file train_data/nl2sql_dev_data.json \
    --test_file train_data/nl2sql_dev_data.json \
    --overwrite_cache \
    --prompt_column question \
    --response_column answer \
    --model_name_or_path ../../../models/chatglm3-6b/ \
    --ptuning_checkpoint output/$CHECKPOINT/checkpoint-600/ \
    --output_dir ./output/$CHECKPOINT \
    --overwrite_output_dir \
    --max_source_length 2200 \
    --max_target_length 300 \
    --per_device_eval_batch_size 1 \
    --predict_with_generate \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 8
