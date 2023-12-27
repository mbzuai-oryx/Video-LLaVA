import os
import argparse
import json
from tqdm import tqdm
from video_chatgpt.eval.model_utils import initialize_model, load_video
from video_chatgpt.inference import video_chatgpt_infer
import sys

from grounding_new_api import Tracker_with_GroundingDINO
from grounding_new_api import cfg as default_cfg 

from deva.utils.pano_utils import ID2RGBConverter
id2rgb_converter = ID2RGBConverter()

from PIL import Image
import torch
import numpy as np
import torchvision
import supervision as sv
import cv2

############################################################

import openai
import ast

openai.api_base = "http://localhost:8000/v1" 
openai.api_key = "EMPTY"  
openai_model_name = 'vicuna-13b-v1.5'

def annotate(question, answer):
    """
    Evaluates question and answer pairs using GPT-3
    Returns a noun phrase (single word) and a referring expression.
    """
    try:
        # Compute the noun phrase and referring expression
        completion = openai.ChatCompletion.create(
            model=openai_model_name,
            messages=[
                {
                    "role": "system",
                    "content":
                        "You are an intelligent chatbot designed for identifying most relevant subject/object phrases in video-based question-sentence pairs. "
                        # Your task is to compare the answer with the question "
                        # "and determine the most relevant single-word noun phrase and generate a referring expression for it. "
                        # "Here's how you can accomplish the task:"
                        # "------"
                        # "##INSTRUCTIONS: "
                        # "- The predicted answer must be a single-word noun phrase and a detailed referring expression."
                },
                {
                    "role": "user",
                    "content":
                        "Your task is to compare the question with the sentence, and extract the subject or object phrase of the sentence that most accurately answers the given question."
                        "The selected phrase should be short and should contain only one noun."
                        "The selected phrase can include adjectives that explain the attributes of the subject/object."
                        "The selected phrase should not exceed 4 words."
                        "The selected phrase should not include articles ('a', 'the', 'and')."
                        
                        "Please generate the response in the form of a Python dictionary string with keys 'OBJECT', where its value is the extracted phrase in Python string format."
                        "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary. "
                        "For example, your response should look like this: {'OBJECT': 'green toy'}."

                        "Please process the following video-based question-answer pair:\n\n"
                        "Question: who is in front of the guitar at the show? \n"
                        "Answer: A woman in a black dress is in front of the guitar on stage. \n\n"
                        
                },
                {
                    "role": "assistant",
                    "content":
                        "{'OBJECT': 'woman in black dress'}"
                },
                {
                    "role": "user",
                    "content":
                        "Question: who points to the window? \n"
                        "Answer: The old man is pointing to window. \n\n"
                        
                },
                {
                    "role": "assistant",
                    "content":
                        "{'OBJECT': 'old man'}"
                        
                },
                {
                    "role": "user",
                    "content":
                        "Question: who is inside the blue car? \n"
                        "Answer: The driver of the blue car. \n\n"
                        
                },
                {
                    "role": "assistant",
                    "content":
                        "{'OBJECT': 'driver'}"
                        
                },
                {
                    "role": "user",
                    "content":
                        "Please process the following video-based question-answer pair:\n\n"
                        f"Question: {question}\n"
                        f"Answer: {answer}\n\n"
                        
                }
            ]
        )
        # Convert response to a Python dictionary.
        response_message = completion["choices"][0]["message"]["content"]
        response_dict = ast.literal_eval(response_message)
        return response_dict
    except Exception as e:
        print(e)
        return None

############################################################

from util.box_ops import np_box_iou

def iou_per_video(tracking_results, video_level_ann):
    img2box = video_level_ann['img2box']


    sIoU=0

    for idx in video_level_ann['inter_idx']:
        
        # print('idx:', idx)
        mask = tracking_results[idx]['mask']
        tmp_id_to_obj = tracking_results[idx]['tmp_id_to_obj']
        obj_to_tmp_id = tracking_results[idx]['obj_to_tmp_id']
        segments_info = tracking_results[idx]['segments_info']
        all_obj_ids = tracking_results[idx]['all_obj_ids']
        image_np = tracking_results[idx]['image_np']
        prompts = tracking_results[idx]['prompts']
        
        # remap indices
        new_mask = torch.zeros_like(mask)
        for tmp_id, obj in tmp_id_to_obj.items():
            new_mask[mask == tmp_id] = obj.id
        mask = new_mask
        
        for seg in segments_info:
            area = int((mask == seg['id']).sum())
            seg['area'] = area
        # filter out zero-area segments
        segments_info = [s for s in segments_info if s['area'] > 0]

        # 
        out_mask = mask.numpy().astype(np.uint32)
        rgb_mask = np.zeros((*out_mask.shape[-2:], 3), dtype=np.uint8)
        for id in all_obj_ids:
            colored_mask = id2rgb_converter._id_to_rgb(id)
            obj_mask = (out_mask == id)
            rgb_mask[obj_mask] = colored_mask
        
        # draw bounding boxes for the prompts
        all_masks = []
        all_cat_ids = []
        all_scores = []
        for seg in segments_info:
            all_masks.append(mask == seg['id'])
            all_cat_ids.append(seg['category_id'])
            all_scores.append(seg['score'])
        if len(all_masks) > 0:
            all_masks = torch.stack(all_masks, dim=0)
            xyxy = torchvision.ops.masks_to_boxes(all_masks)
            xyxy = xyxy.numpy()
            pred_boxes = xyxy

            ## frame_wise IoU calculation
            # gt_boxes = img2box[image_id]
            # gt_boxes = img2box[list(img2box.keys())[idx]]
            gt_boxes = img2box[video_level_ann['inter_idx_to_inter_frames_map'][idx]]
            # iou = np_box_iou(np.array(pred_boxes), np.array(gt_boxes))[0][0]
            iou = np_box_iou(np.array(pred_boxes), np.array(gt_boxes))
            # print('pred_boxes:', pred_boxes, '  gt_boxes:', gt_boxes)
            # iou = iou[0][0] 
            iou = iou.max()
        else:
            iou = 0
        sIoU +=iou
    sIoU = sIoU /len(video_level_ann['inter_idx'])

    print('sIoU:', sIoU)
    
    return sIoU
    
def get_visualization_for_frame(tracking_results, frame_idx):
    mask = tracking_results[frame_idx]['mask']
    tmp_id_to_obj = tracking_results[frame_idx]['tmp_id_to_obj']
    obj_to_tmp_id = tracking_results[frame_idx]['obj_to_tmp_id']
    segments_info = tracking_results[frame_idx]['segments_info']
    all_obj_ids = tracking_results[frame_idx]['all_obj_ids']
    image_np = tracking_results[frame_idx]['image_np']
    prompts = tracking_results[frame_idx]['prompts']
    
    # remap indices
    need_remapping = True

    if need_remapping:
        new_mask = torch.zeros_like(mask)
        for tmp_id, obj in tmp_id_to_obj.items():
            new_mask[mask == tmp_id] = obj.id
        mask = new_mask
        
    # record output in the json file
    # if saver.json_style == 'vipseg': #NOTE:vigsepg format
    for seg in segments_info:
        area = int((mask == seg['id']).sum())
        seg['area'] = area
    # filter out zero-area segments
    segments_info = [s for s in segments_info if s['area'] > 0]
    # append to video level storage
    this_annotation = {
        'file_name': 'null' + '.jpg',
        'segments_info': segments_info,
    }
    
    use_long_id = True
    if use_long_id:
        out_mask = mask.numpy().astype(np.uint32)
        rgb_mask = np.zeros((*out_mask.shape[-2:], 3), dtype=np.uint8)
        for id in all_obj_ids:
            colored_mask = id2rgb_converter._id_to_rgb(id)
            obj_mask = (out_mask == id)
            rgb_mask[obj_mask] = colored_mask
        out_img = Image.fromarray(rgb_mask)
    
    # 
    alpha = (out_mask == 0).astype(np.float32) * 0.5 + 0.5
    alpha = alpha[:, :, None]
    blend = (image_np * alpha + rgb_mask * (1 - alpha)).astype(np.uint8)

    if prompts is not None:
        # draw bounding boxes for the prompts
        all_masks = []
        labels = []
        all_cat_ids = []
        all_scores = []
        for seg in segments_info:
            all_masks.append(mask == seg['id'])
            labels.append(f'{prompts[seg["category_id"]]} {seg["score"]:.2f}')
            all_cat_ids.append(seg['category_id'])
            all_scores.append(seg['score'])
        if len(all_masks) > 0:
            all_masks = torch.stack(all_masks, dim=0)
            xyxy = torchvision.ops.masks_to_boxes(all_masks)
            xyxy = xyxy.numpy()

            detections = sv.Detections(xyxy,
                                        confidence=np.array(all_scores),
                                        class_id=np.array(all_cat_ids))
            annotator = sv.BoxAnnotator()
            blend = annotator.annotate(scene=blend,
                                        detections=detections,
                                        labels=labels)
            
    
    return this_annotation, blend

############################################################

def save_videos(output_dir, image_frames, targets_list_2, tracking_results, idx, fps=5, video_caption=''):
    num_frames = image_frames.shape[0]

    # save ground truth    
    output_path_gt = os.path.join(output_dir, 'ground_truth', f'{idx}.mp4')
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
            
            # 
            writer_gt.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
            
        else:
            writer_gt.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer_gt.release()
    
    # save predictions
    output_path_pred = os.path.join(output_dir, 'predictions', f'{idx}.mp4')
    writer_pred = None
    for i in range(len(tracking_results)):
        _, blend = get_visualization_for_frame(tracking_results, frame_idx=i)
        res_ = cv2.cvtColor(blend, cv2.COLOR_RGB2BGR)
        if not writer_pred:
            writer_pred = cv2.VideoWriter(output_path_pred, cv2.VideoWriter_fourcc(*'MP4V'), fps, (res_.shape[1],res_.shape[0]))
        writer_pred.write(res_)
    writer_pred.release()    


def save_chat_results(output_dir, idx, qtype, question, llm_output, response_dict, sIoU):
    output_json_path = os.path.join(output_dir, 'predictions', f'{idx}.json')
    
    # Data to be written
    json_dictionary = {
        "qtype": qtype,
        "question": question,
        "llm_output": llm_output,
        "response_dict": response_dict,
        "sIoU":sIoU
    }
    
    # Serializing json
    json_object = json.dumps(json_dictionary, indent=4)
    
    # Writing to sample.json
    with open(output_json_path, "w") as outfile:
        outfile.write(json_object)
    
############################################################
from pathlib import Path

def get_dataset(dataset_name, vid_dir, ann_dir, fps):
    if dataset_name=='hcstvg':
        from datasets.hcstvg_dataset import HCSTVG_Dataset
        
        vid_dir = Path(vid_dir)
        # using 'test' set
        test = True 
        image_set = 'test'
        v2 = True
        if test or image_set == "val":
            if not v2:  # only a test set
                ann_file = Path(ann_dir) / f"test_proc.json"
            else:  # only a val set
                ann_file = Path(ann_dir) / f"val_v2_proc.json"
        else:
            if not v2:
                ann_file = Path(ann_dir) / f"train_proc.json"
            else:
                ann_file = Path(ann_dir) / f"train_v2_proc.json"
        

        hcstvg_dataset = HCSTVG_Dataset(
            vid_dir,
            ann_file,
            image_set=image_set,
            video_max_len=100,
            required_fps=fps,
            take_only_temp_loc_frames=True,
            resolution=args.resolution
        )
        return hcstvg_dataset
    
    if dataset_name=='vidstg':
        from datasets.vidstg_dataset import VidSTG_Val_Test
        
        vidstg_ann_path = ann_dir
        vidstg_vid_path = vid_dir 

        # used vidstg-test set with only temporally localized frames
        vidstg_dataset = VidSTG_Val_Test(vidstg_ann_path,vidstg_vid_path, take_only_temp_loc_frames = True, image_set='test', required_fps=fps, video_max_len=100, resolution=args.resolution)
        
        return vidstg_dataset
    
def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    
    parser.add_argument("--model", type=str, required=True, choices=['gdino_baseline', 'video_chatgpt'])
    
    parser.add_argument("--model-name", type=str, required=False)
    parser.add_argument("--projection_path", type=str, required=False)
    parser.add_argument("--conv_mode", type=str, required=False, default='pg-video-llava')
    
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--resolution", type=int, required=True)
    
    parser.add_argument("--dataset", type=str, choices=['vidstg', 'hcstvg'])
    parser.add_argument("--vid_dir", type=str, required=True)
    parser.add_argument("--ann_dir", type=str, required=True)
    parser.add_argument("--hcstvg_qa_dir", type=str, required=False)
    
    args = parser.parse_args()
    return args

if __name__=='__main__':

    # Parse args
    args = parse_args()
    
    ##
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(os.path.join(args.output_dir, 'ground_truth')):
        os.makedirs(os.path.join(args.output_dir, 'ground_truth'))
    if not os.path.exists(os.path.join(args.output_dir, 'predictions')):
        os.makedirs(os.path.join(args.output_dir, 'predictions'))

    
    ##
    if args.model=='video_chatgpt':
        model, vision_tower, tokenizer, image_processor, video_token_len = initialize_model(args.model_name, args.projection_path)

    tracker = Tracker_with_GroundingDINO(config=default_cfg, deva_model_path=default_cfg['deva_model_path'], 
                                         temporal_setting='online',
                                         detection_every=2,
                                         max_missed_detection_count=1,
                                         max_num_objects=1,
                                         dino_threshold=0.35)


    ## Dataset
    VIDEO_FPS=5
    dataset = get_dataset(args.dataset, args.vid_dir, args.ann_dir , VIDEO_FPS)
    print('len(dataset): ', len(dataset))
    
    ##
    m_sIoU=0
    num_interrogative_captions=0
    num_finished_processing=0

    for idx in tqdm(range(len(dataset))):
    
        if args.dataset=='hcstvg':
            try:
                f = open( os.path.join(args.hcstvg_qa_dir, f"{idx}.json"))
                res_dict = json.load(f)
                f.close()
                            
                question, answer = res_dict['Q'], res_dict['A']
                assert not(question=='') and not answer==''
            except FileNotFoundError as e:
                print(f'{idx}.json not found. Skipping ...')
                continue
            image_frames_np, targets, _ , _ , video_level_ann, targets_list_2 = dataset[idx]
            image_frames_pil = [Image.fromarray(frame) for frame in image_frames_np]

            num_interrogative_captions+=1
            qtype = 'interrogative'
            
        if args.dataset=='vidstg':
            image_frames_np, targets, qtype, video_caption, video_level_ann, targets_list_2 = dataset[idx]
            image_frames_pil = [Image.fromarray(frame) for frame in image_frames_np]

            if qtype=='declarative':
                continue
            else: #'interrogative'
                question = video_caption
                num_interrogative_captions+=1
    
        print('question:', question)

        if args.model=='video_chatgpt':
            # Run inference on the video
            with torch.no_grad():
                prompt_to_video_chatgpt = f"QUESTION: {question}\n\n" + "Please provide your answer to the QUESTION in one sentence."
                llm_output = video_chatgpt_infer(image_frames_pil, prompt_to_video_chatgpt, args.conv_mode, model, vision_tower, tokenizer, image_processor, video_token_len)
            print('llm_output:', llm_output, "\n\n")
            response_dict = annotate(question, llm_output)
            print('response_dict:', response_dict)
            try:
                referring_exp = response_dict['OBJECT']
                if isinstance(referring_exp, list):
                    referring_exp = referring_exp[0]
            except Exception as e:
                print(e)
                continue
            print('referring_exp:', referring_exp)
        
        else:
            referring_exp=question
            llm_output=''
            response_dict={}
        
        try:
            class_list = [referring_exp] # because there is only one class at a time
            tracking_results = tracker.run_on_list_of_images(image_frames_pil,class_list=class_list)
            # calculate sIoU
            sIoU = iou_per_video(tracking_results, video_level_ann)
            m_sIoU+=sIoU
        except Exception as e:
            print(e)
            continue


        # save videos
        save_videos(args.output_dir, image_frames_np, targets_list_2, tracking_results, idx, fps=VIDEO_FPS, video_caption=question)
        save_chat_results(args.output_dir, idx, qtype, question, llm_output, response_dict, sIoU)
        num_finished_processing+=1
        print(f'\nidx:{idx}    num_interrogative_captions:{num_interrogative_captions}  num_finished_processing:{num_finished_processing}   Running m_sIoU: ', m_sIoU/num_interrogative_captions,'\n\n')

    m_sIoU = m_sIoU/num_interrogative_captions
    print(f'Processed {num_interrogative_captions} videos')
    print('m_sIoU:', m_sIoU)
