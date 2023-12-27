# Quantitative Evaluation of Conversation-based Video Spatial Grounding 

We introduce novel benchmarks for quantitatively evaluating conversation-based video spatial grounding, based on two exisiting spatio-temporal video grounding datasets, VidSTG and HC-STVG. 


In conversation-based spatial grounding, the objective is to localise interrogative sentences with unknown objects in the given video (e.g. ”What is caught by the squatting boy on the floor?” ). Unlike grounding for declarative sentences where the explicit characteristics of objects (e.g. the class ”toy” and visual appearance ”yellow”) are present within the sentence itself, grounding for interrogative sentences is challenging due to the fact that it can only depend on relationships between the unknown object and other objects (e.g. the action relation ”caught by the squatting boy” and spatial relation ”on the floor”). A benchmark based on this task can be regarded a measure for the sufficient relationship construction and crossmodal relation reasoning ability of the video-language model.

To evaluate our model for conversation-based video spatial grounding, we pass interrogative prompts to the model. It then generates descriptive textual responses to these prompts, from which Vicuna-13b-v1.5 extracts relevant referring expressions. These expressions are then passed into the GroundingDINO-based spatial grounding and tracking module. For obtained object tracks, bounding box based IoU is calculated by comparing them with the ground truth annotations. 

From the two spatio-temporal grounding datasets, to form a spatial-only grounding benchmark, we crop the video in the temporal axis to contain only the segment where the target object is present, and the mean spatial IoU is reported as the metric for comparison.

It should be noted that we evaluate our model in these benchmarks only in the zero-shot setting, without any training on these datasets.

## 1. Benchmark based on VidSTG Dataset

VidSTG datasets consists of videos paired with multiform sentences (both interrogative and declarative). To form a benchmark to quantitatively evaluate the performance of conversation-based video spatial grounding, we leverage the 5693 video and interrogative sentence pairs in its test set. 

**Steps to reporduce the benchmark results with our model**:

1. Download the VidSTG dataset from [here](https://github.com/Guaranteer/VidSTG-Dataset). The recommended folder structure is:

        data
        └── VidSTG
            └── video
            |   ├── 0000
            |   └── 0001
            |   └── ....
            |   └── ....
            └── vidor_annotations
            |   ├── training
            |   └── validation
            └── vistg_annotations
                ├── train_files.json
                └── val_files.json 
                ├── test_files.json 
                └── train_annotations.json
                ├── val_annotations.json
                └── test_annotations.json
                
   
2. Preprocess the dataset
            
        python grounding_evaluation/datasets/preproc_vidstg.py \
            --vidor_annotations_dir <vidor_annotations_dir> \
            --vidstg_annotations_dir <vidstg_annotations_dir>

3. Run evaluation script

        python grounding_evaluation/eval_grounding.py \
            --model video_chatgpt \
            --model-name path/to/llava-v1.5-13b \
            --projection_path path/to/mm_projector.bin \
            --output_dir <your_output_directory> \
            --resolution 336 \
            --dataset vidstg \
            --vid_dir <vidstg_video_directory>  \
            --ann_dir <vidstg_annotation_directory>


## 2. Benchmark based on HCSTVG Dataset

Unlike in VidSTG dataset, HC-STVG dataset contains only declarative form sentences for all of its videos. Therefore interrogative sentences are first generated from the declarative text captions in 3025 samples of the test set using Vicuna-13b-v1.5 model. Then the evaluation is performed in a siminlar manner to VidSTG. 

**Steps to reporduce the benchmark results with our model**:

1. Download the HC-STVG dataset (V2) from [here](https://github.com/tzhhhh123/HC-STVG). Download the extracted question-answer pairs from [here](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/shehan_munasinghe_mbzuai_ac_ae/EpfIekgGqSNDniItu5WxaOMBJt8fFMldSrcHwg_sTieI6w?e=818VNL). The recommended folder structure is:

        data
        └── HC-STVG
            └── qa
            |   ├── 0.json
            |   └── 1.json
            |   └── ....
            |   └── ....
            └── Video
            |   ├── 0
            |   ├── 1
            |   └── ...
            └── anno_v2
                ├── query_v2.json
                ├── train_v2.json 
                └── val_v2.json 

2. Preprocess the dataset

        python grounding_evaluation/datasets/preproc_hcstvgv2.py \
            --video_dir <hcstvg_video_dir> \
            --ann_dir <hcstvg_annotations_dir>

3. Running Evaluation

        python grounding_evaluation/eval_grounding.py \
            --model video_chatgpt \
            --model-name path/to/llava-v1.5-13b \
            --projection_path path/to/mm_projector.bin \
            --output_dir <your_output_directory> \
            --resolution 336 \
            --dataset hcstvg \
            --vid_dir <hcstvg_video_directory>  \
            --ann_dir <hcstvg_annotation_directory> \
            --hcstvg_qa_dir <hcstvg_qa_directory>

        