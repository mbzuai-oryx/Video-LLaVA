import tempfile
import argparse

from grounding_new_api import Tracker_with_GroundingDINO
from grounding_new_api import cfg as default_cfg 

import os
import argparse
import json
from tqdm import tqdm
import subprocess
import glob

from video_chatgpt.eval.model_utils import initialize_model, load_video
from video_chatgpt.inference import video_chatgpt_infer
from video_chatgpt.audio_transcript.transcribe import Transcriber

from util.image_tagging import TaggingModule, get_unique_tags

############################################################

    
def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    
    # parser.add_argument("--model", type=str, required=True, choices=['gdino_baseline', 'video_chatgpt'])
    
    parser.add_argument("--model-name", type=str, required=False)
    parser.add_argument("--projection_path", type=str, required=False)
    parser.add_argument("--conv_mode", type=str, required=False, default='pg-video-llava')
    parser.add_argument("--use_asr", action='store_true', help='Whether to use audio transcripts or not')
    
    parser.add_argument("--input_video_path", type=str, required=True)
    parser.add_argument("--output_video_path", type=str, required=True)
    
    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = parse_args()
    
    # with tempfile.TemporaryDirectory() as tmpdirname:
    temp_dir_splits = tempfile.TemporaryDirectory()
    temp_dir_saves = tempfile.TemporaryDirectory()
    # print('temp_dir_splits: ', temp_dir_splits,'    temp_dir_saves: ', temp_dir_saves)
    # print(temp_dir_splits.name)
    
    
    # Model
    model, vision_tower, tokenizer, image_processor, video_token_len = initialize_model(args.model_name, args.projection_path)
    conv_mode = args.conv_mode
    frame_size = (image_processor.crop_size['height'], image_processor.crop_size['width'])
    if args.use_asr:
        transcript_model = Transcriber()
        
    # Inference
    video_frames = load_video(args.input_video_path, shape=frame_size)
    if args.use_asr:
        transcript_text = transcript_model.transcribe_video(video_path=args.input_video_path)
    else:
        transcript_text=None
    
    question = input("Enter a question about the video: ") 
    
    # Run inference on the video and add the output to the list
    llm_output = video_chatgpt_infer(video_frames, question, conv_mode, model, vision_tower,
                                        tokenizer, image_processor, video_token_len, transcript_text)
    
    print("\n\n", llm_output)
    

    tracker = Tracker_with_GroundingDINO(config=default_cfg, deva_model_path=default_cfg['deva_model_path'], 
                                         temporal_setting='online',
                                        #  detection_every=2,
                                        #  max_missed_detection_count=1,
                                        #  max_num_objects=1, #TODO change
                                        #  dino_threshold=0.35
                                        )
    
    tagging_model = TaggingModule()
    
    from util.entity_matching_openai import EntityMatchingModule

    entity_match_module = EntityMatchingModule()
    
    tags_in_video = tagging_model.run_on_video(video_frames)
    classes = get_unique_tags(tags_in_video)
    entity_list = classes[:10]
        

    highlight_output, match_state = entity_match_module(llm_output, entity_list)
    class_list = list(set(match_state.values()))
    
    # Split video
    _ = subprocess.call(["scenedetect", "-i", args.input_video_path, 
                        "split-video", "-o", temp_dir_splits.name
                        ]) # scenedetect -i {args.input_video_path} split-video -o {temp_dir_splits.name}

    
    # For each split
    for video_split_path in os.listdir(temp_dir_splits.name):

        tracker.run_on_video(os.path.join(temp_dir_splits.name, video_split_path), os.path.join(temp_dir_saves.name,video_split_path), class_list)


    # Combine splits and save
    mp4_files = sorted(glob.glob(os.path.join(temp_dir_saves.name, '*.mp4')))
    txt = ''
    with open(os.path.join(temp_dir_saves.name, 'video_list.txt'), 'w') as file:
        for mp4_file in mp4_files:
            txt+= "file "+ mp4_file + '\n'
        file.write(txt)
    cmd_to_combine_splits_2 = f"ffmpeg -f concat -safe 0 -i {temp_dir_saves.name}/video_list.txt -c copy {args.output_video_path} -y"
    cmd_to_combine_splits_3 = f"rm {temp_dir_saves.name}/video_list.txt"

    # _ = subprocess.run([cmd_to_combine_splits_1], shell=True)
    _ = subprocess.run([cmd_to_combine_splits_2], shell=True)
    _ = subprocess.run([cmd_to_combine_splits_3], shell=True)

    # use temp_dir_splits, and when done:
    temp_dir_splits.cleanup()
    temp_dir_saves.cleanup()