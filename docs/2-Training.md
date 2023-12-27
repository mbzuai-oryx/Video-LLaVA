# Training Our Model

We train our model on the [VideoInstruct100K dataset](https://huggingface.co/datasets/MBZUAI/VideoInstruct-100K).

Steps: 

1. Download our 100K video instruction dataset from [here](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/hanoona_bangalath_mbzuai_ac_ae/EWxYslvDeX1PijKWM_WxTkkBDXDDD350YnUQOkbcL8V7Xg?e=Lq9itD).
<!-- 2. Convert the downloaded JSON into the required format for training.

        python scripts/convert_instruction_json_to_training_format.py \
            --input_json_file <path to json file downloaded in step 2> \
            --output_json_file video_chatgpt_training.json

    The above script will generate *video_chatgpt_training.json* required to train our model. -->

2. Download ActivityNet videos. 
   
    All the videos annotated in our work are taken from ActivityNet dataset. You can download these from [here](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/hanoona_bangalath_mbzuai_ac_ae/EnLRDehrr8lGqHpC5w1zZ9QBnsiVffYy5vCv8Hl14deRcg?e=Ul5DUE).

3. Prepare Spatio-Temporal features using CLIP Note that for training efficiency, we pre-computed the video spatio-temporal features and use them directly during training. After downloading the videos, please use the following command to generate CLIP spatio-temporal features.

        python scripts/save_spatio_temporal_clip_features.py \
            --video_dir_path <path to the directory containing all the videos> \
            --clip_feat_path <The output dir to save the features in.>
    
    The script will generate the spatiotempral features for each video and save one pickle file per video in directory specified by --clip_feat_path argemunt.

4. Convert the downloaded JSON into the required format for training.


        python scripts/filter_for_missing_videos.py \
            --input_json_file <path to VideoInstruct_Dataset.json donwloaded in step 1 >  \
            --clip_feature_path <clip_feat_path in step 3> \
            --output_json_file <video_chatgpt_training.json> 
            

    The above script will generate *video_chatgpt_training.json* required to train our model.


5. Train the model. 
   
        torchrun --nproc_per_node=4 --master_port 29001 video_chatgpt/train/train_mem.py \
            --model_name_or_path <path to LLaVA-v1.5 model> \
            --version v1 \
            --data_path <path to filtered_336_video_chatgpt_training.json> \
            --video_folder <path to the spatio-temporal features generated in step 4 using `save_spatio_temporal_clip_features.py` script> \
            --tune_mm_mlp_adapter True \
            --mm_use_vid_start_end \
            --bf16 True \
            --output_dir <Video-ChatGPT_13B-1.1_Checkpoints_with_LLaVA-1.5_and_336px> \
            --num_train_epochs 3 \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 8 \
            --gradient_accumulation_steps 1 \
            --evaluation_strategy "no" \
            --save_strategy "steps" \
            --save_steps 3000 \
            --save_total_limit 3 \
            --learning_rate 2e-5 \
            --weight_decay 0. \
            --warmup_ratio 0.03 \
            --lr_scheduler_type "cosine" \
            --logging_steps 100 \
            --tf32 True \
            --model_max_length 2048 \
            --gradient_checkpointing True \
            --lazy_preprocess True
        