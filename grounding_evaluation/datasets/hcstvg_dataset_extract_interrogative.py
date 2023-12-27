import os
import json
from pathlib import Path
import time
import numpy as np
import random
from PIL import Image
import copy
from tqdm import tqdm

from hcstvg_dataset import HCSTVG_Dataset

import argparse

if __name__=='__main__':

    # hcstvg_ann_path = "/share/users/shehan/data/HC-STVG/anno_v2"
    # hcstvg_vid_path = "/share/users/shehan/data/HC-STVG/Video"
    
    parser = argparse.ArgumentParser(description="Extract Question-Answer Pairs from HC-STVG Dataset")
    parser.add_argument("--hcstvg_vid_dir", required=True, help="The path to the directory containing HC-STVG videos")
    parser.add_argument("--hcstvg_ann_dir", required=True, help="The path to the directory containing HC-STVG annotations")
    args = parser.parse_args()
    
    hcstvg_vid_path = args.hcstvg_vid_dir
    hcstvg_ann_path = args.hcstvg_ann_dir

    vid_dir = Path(hcstvg_vid_path)

    # using 'test' set
    test = True 
    image_set = 'test'
    v2 = True

    if test or image_set == "val":
        if not v2:  # only a test set
            ann_file = Path(hcstvg_ann_path) / f"test_proc.json"
        else:  # only a val set
            ann_file = Path(hcstvg_ann_path) / f"val_v2_proc.json"
    else:
        if not v2:
            ann_file = Path(hcstvg_ann_path) / f"train_proc.json"
        else:
            ann_file = Path(hcstvg_ann_path) / f"train_v2_proc.json"
            
    VIDEO_FPS=5

    dataset = HCSTVG_Dataset(
        vid_dir,
        ann_file,
        image_set=image_set,
        video_max_len=100,
        required_fps=VIDEO_FPS,
        take_only_temp_loc_frames=True,
        resolution=224
    )


    import openai
    import ast

    openai.api_base = "http://localhost:8000/v1" 
    openai.api_key = "EMPTY"  
    openai_model_name = 'vicuna-13b-v1.5'

    def annotate(caption):
        """
        Generate question and answer pairs using GPT-3
        Returns a question and the corresponding answer.
        """
        try:
            # Compute the noun phrase and referring expression
            completion = openai.ChatCompletion.create(
                model=openai_model_name,
                messages=[
                    {
                        "role": "system",
                        "content":
                            "You are an intelligent chatbot designed for generating question-answer pairs from sentences."
                    },
                    {
                        "role": "user",
                        "content":
                            "Your task is to generate a question and and answer from the given sentence."
                            "The question should start with 'Who'."
                            "The question should refer to the subject of the given sentece."
                            "The answer should include the subject of the given sentence."
                            "Please generate the response in the form of a Python dictionary string with keys 'Q' for question and 'A' for answer. Each corresponding value should be the question and answer text respectively."
                            "For example, your response should look like this: {'Q': 'Your question here...', 'A': 'Your answer here...'}. "
                            "Please note that the generated question and answer should only include information from the given sentence."
                            "\n"
                            "Please process the following sentence : \n The man in the suit goes to the man in white and looks at him."
                            
                    },
                    {
                        "role": "assistant",
                        "content":
                            "{'Q': 'Who goes to the man in white?', 'A':'The man in the suit'}"
                            
                            
                    },
                    {
                        "role": "user",
                        "content":
                            f"Please process the following sentence : \n{caption}."
                            
                    },
                ]
            )
            # Convert response to a Python dictionary.
            response_message = completion["choices"][0]["message"]["content"]
            # print(response_message)
            response_dict = ast.literal_eval(response_message)
            return response_dict
        except Exception as e:
            print(e)
            return None

    import cv2
    import supervision as sv

    def save_videos(image_frames, targets_list_2, video_caption, fps=5, output_dir='hcstvg', idx=''):
        num_frames = image_frames.shape[0]

        # save ground truth    
        output_path_gt = os.path.join(output_dir, f'{idx}.mp4')
        h, w = image_frames[0].shape[:2]
        writer_gt = cv2.VideoWriter(output_path_gt, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        
        for t in range(num_frames):
            frame = image_frames[t]

            traj_boxes_in_frame = [obj["bbox"] for obj in targets_list_2[t]]
            
            if traj_boxes_in_frame: # if  detections in the frame 
                
                traj_boxes_in_frame = np.array(traj_boxes_in_frame)
                detections = sv.Detections(traj_boxes_in_frame)

                box_annotator = sv.BoxAnnotator()
                
                frame_rgb= frame 
                annotated_frame = box_annotator.annotate(
                    scene=frame_rgb,
                    detections=detections,
                    labels=[video_caption]
                )
                
                writer_gt.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
                
            else:
                writer_gt.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer_gt.release()


    for idx in tqdm(range(len(dataset))):
        image_frames_np, targets, qtype, video_caption, video_level_ann, targets_list_2 = dataset[idx]
        
        res_dict = annotate(video_caption)
        
        try:
            if res_dict and 'Q' in res_dict.keys():
                save_videos(image_frames_np, targets_list_2, res_dict['Q'], fps=VIDEO_FPS, output_dir='hcstvg/gt', idx=idx)
                
                with open(f"hcstvg/qa/{idx}.json", "w") as outfile:
                    json.dump(res_dict, outfile)
        except Exception as e:
            print(e)
            continue    
