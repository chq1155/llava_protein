python llava/train/train_mem.py \
    --model_name_or_path /root/llava_download/vicuna/checkpoints/llava-7b-pretrain \
    --version v1 \
    --data_path /root/llava_download/cc3m_595k/chat.json \
    --image_folder /root/llava_download/vicuna/image \
    --vision_tower /root/llava_download/vision_encoder \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end \
    --bf16 True \
    --output_dir /root/llava_download/vicuna_pretrain/output \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2400 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to wandb