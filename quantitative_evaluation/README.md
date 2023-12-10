# Quantitative Evaluation Framework for Video-based Conversational Models

The major modification we made to the benchmark introduced in Video-ChatGPT for video-based conversational models is utilizing open-source Vicuna-13B-v1.5 model instead of  OpenAI GPT-3.5-Turbo. The main motivation to do this is to ensure the reproducibility of results which is a concern with the proprietary nature of GPT-3.5.

### Steps to setup OpenAI-compatible FastChat API to serve Vicuna-13B-v1.5 locally:

1. Install FastChat as mentioned [here](https://github.com/lm-sys/FastChat/tree/main). (For more details about the API, look [here](https://github.com/lm-sys/FastChat/blob/main/docs/openai_api.md).)

2. Run the following commands to serve Vicuna-13B-v1.5 model

        python3 -m fastchat.serve.controller
        python3 -m fastchat.serve.model_worker --model-path lmsys/vicuna-13b-v1.5
        python3 -m fastchat.serve.openai_api_server --host localhost --port 8000
    
        export OPENAI_API_BASE=http://localhost:8000/v1
        export OPENAI_API_KEY=EMPTY


## 1. Video-based Generative Performance Benchmarking

In this benchmark we continue to use the same test set of 500 samples curated from the ActivityNet-200 videos as in Video-ChatGPT. The videos can be downloaded from [here](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/hanoona_bangalath_mbzuai_ac_ae/EatOpE7j68tLm2XAd0u6b8ABGGdVAwLMN6rqlDGM_DwhVA?e=90WIuW) and the corresponding question-answer pairs are available from [here](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/hanoona_bangalath_mbzuai_ac_ae/EoS-mdm-KchDqCVbGv8v-9IB_ZZNXtcYAHtyvI06PqbF_A?e=1sNbaa).

Follow the steps below to perform the quantitative benchmarking:

**Step 1**: Run the inference using the provided question-answer pairs for each criteria.

    python video_chatgpt/eval/run_inference_benchmark_general.py \
        --video_dir <path-to-directory-containing-videos> \
        --gt_file <ground-truth-file-containing-question-answer-pairs> \
        --output_dir <output-dir-path> \
        --output_name <output-file-name> \
        --model-name <path-to-LLaVA-v1.5-7B> \
        --projection_path <path-to-projector-weights-for-LLaVA-v1.5-7B>

* Note that the question-answer pairs (gt_file) for evaluating correctness, detailed orientation and Contextual understanding are the same (`generic_qa.json`). For temporal understanding and consistency, separate question-answer pairs are provided (`temporal_qa.json` and `consistency_qa.json`).


**Step 2**: Execute the corresponding evaluation script to perform benchmarking.

For example, for correctness criteria:

    python quantitative_evaluation/evaluate_benchmark_1_correctness.py \
        --pred_path <path-to-prediction-file-generated-using-inference-script> \
        --output_dir <output-directory-path> \
        --output_json <path-to-save-annotation-final-combined-json-file> \
        --model_name "vicuna-13b-v1.5" \
        --api_base "http://localhost:8000/v1" \
        --api_key "EMPTY"
    
For evaluation on all 5 criteria, you can use:

    bash quantitative_evaluation/evaluate_benchmark.sh


## 2. Zero-Shot Question-Answer Evaluation

Following Video-ChatGPT, we perform zero-shot evaluation on four standard open-ended question-answer datasets: MSRVTT, MSVD, TGIF, and ActivityNet-QA.

Here we present the evaluation method on ActivityNet-QA. The evaluation protocol remains the same for all datasets, except for some dataset-specific changes related to videos and annotations.

### E.g.: Evaluation on ActivityNet-QA

**Step 1**: Run the inference. 

* You'll need the following:

  * Videos: Download the videos for ActivityNet-QA from [here](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/hanoona_bangalath_mbzuai_ac_ae/ESa302OCJMNHsMk7wuBbQc8BZH5CqlcdCWiSpXynQZDfAQ?e=CrOPbm).

  * Question and answer annotations: You can obtain these from the official [GitHub repository](https://github.com/MILVLG/activitynet-qa/tree/master/dataset), or download from [here](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/hanoona_bangalath_mbzuai_ac_ae/El1SR1Mri2NLgptt4jTOy1wBJkGyzXDKGvsWFLxvdbpKPw?e=vxtpNu).



* Run the command:

        python video_chatgpt/eval/run_inference_qa_activitynet.py \
            --video_dir <path-to-video-dir> \
            --gt_file_question <test_q.json> \
            --gt_file_answers <test_a.json> \
            --output_dir <path-to-out-dir> \
            --output_name video_chatgpt_activitynet_qa_preds \
            --model-name <path-to-LLaVA-v1.5-7B> \
            --projection_path <path-to-projector-weights-for-LLaVA-v1.5-7B>

This will generate a JSON file containing the model's predicted responses.


**Step 2**: Evaluate the predicted responses. 

The evaluation process computes the accuracy and assigns a score on a scale of 1-5. This step requires the predictions from step-1 and question-answer pair annotations.

* Run the command:

        python quantitative_evaluation/evaluate_activitynet_qa.py \
            --pred_path <video_chatgpt_activitynet_qa_preds> \
            --output_dir <path-to-out-dir> \
            --output_json <video_chatgpt_activitynet_qa_results> \
            --api_base "http://localhost:8000/v1" \
            --api_key "EMPTY" \
            --model_name "vicuna-13b-v1.5" \
            --num_tasks 1
            