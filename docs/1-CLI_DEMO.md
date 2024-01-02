## Installation

We recommend setting up a conda environment for the project:

    conda create --name=pg_video_llava python=3.10
    conda activate pg_video_llava
    
    git clone https://mbzuai-oryx.github.io/Video-LLaVA/
    cd Video-LLaVA
    pip install -r requirements.txt
    
    export PYTHONPATH="./:$PYTHONPATH"

Additionally, install FlashAttention which will be required for training,

    pip install ninja

    git clone https://github.com/HazyResearch/flash-attention.git
    cd flash-attention
    git checkout v1.0.7
    python setup.py install

<!-- TODO: update requirements.txt -->

## Download PG-Video-LLaVA Weights

* Download LLaVA-v1.5 weights from HuggingFace
  * [LLaVA-v1.5-7B](https://huggingface.co/liuhaotian/llava-v1.5-7b)
  * [LLaVA-v1.5-13B](https://huggingface.co/liuhaotian/llava-v1.5-13b)

* Download projector weights:
  * Projector for LLaVA-v1.5-7B [(download)](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/shehan_munasinghe_mbzuai_ac_ae/ESaOp2uTRF5Al00GOvZKWFsB438GAct1UQME_oqDtv6cUw?e=5vVo6P)
  * Projector for LLaVA-v1.5-13B [(download)](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/shehan_munasinghe_mbzuai_ac_ae/ESaaVskN3NhLhAk-ZsOCg4wBkSNKqJc8dKJtjpvETEL2GA?e=ktjVZG)
 
The reset of the documentation assumes these weight files are stored in the following structure.

    Video-LLaVA
    └── weights
        └── llava
        |   ├── llava-v1.5-7b
        |   └── llava-v1.5-13b
        └── projection
            ├── mm_projector_7b_1.5_336px.bin
            └── mm_projector_13b_1.5_336px.bin

## Download Weights for Grounding Module

* Setup DEVA as mentioned [here](https://github.com/hkchengrex/Tracking-Anything-with-DEVA)
* Setup Grounded-Segment-Anything as mentioned [here](https://github.com/hkchengrex/Grounded-Segment-Anything)
* Save or symlink all the tracker weights at *Video-LLaVA/grounding_evaluation/weights*. (The weight files can be donwloaded from [here](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/shehan_munasinghe_mbzuai_ac_ae/Eog31ej6Ah5BmOYqyHJpXvEBJreSD4ZE_YfJ8H_xELg94Q?e=U47VYU))

        Video-LLaVA
        └── grounding_evaluation
            └── weights
                ├── DEVA-propagation.pth
                └── groundingdino_swint_ogc.pth
                └── GroundingDINO_SwinT_OGC.py
                └── mobile_sam.pt
                └── ram_swin_large_14m.pth
                └── sam_vit_h_4b8939.pth
                


## Download sample videos

<!-- TODO: Upload to OneDrive -->

## Run CLI Demo

E.g.: Run CLI Demo without grounding.

    export PYTHONPATH="./:$PYTHONPATH"
    python video_chatgpt/chat.py \
        --model-name  <path_to_LLaVA-7B-1.5_weights> \
        --projection_path <path_to_projector_wights_for_LLaVA-7B-1.5> \
        --use_asr \
        --conv_mode pg-video-llava

E.g.: Run CLI Demo with grounding.

    export PYTHONPATH="./:$PYTHONPATH"
    export OPENAI_API_KEY=<OpenAI API Key>
    python video_chatgpt/chat.py \
        --model-name  <path_to_LLaVA-7B-1.5_weights> \
        --projection_path <path_to_projector_wights_for_LLaVA-7B-1.5> \
        --use_asr \
        --conv_mode pg-video-llava \
        --with_grounding

