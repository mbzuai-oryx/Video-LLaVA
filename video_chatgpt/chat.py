import argparse
from video_chatgpt.video_conversation import (default_conversation)
from video_chatgpt.video_conversation import load_video
from video_chatgpt.video_conversation import conv_templates, SeparatorStyle
from video_chatgpt.model.utils import KeywordsStoppingCriteria
from video_chatgpt.audio_transcript.transcribe import Transcriber

from video_chatgpt.utils import disable_torch_init
from video_chatgpt.eval.model_utils import initialize_model
from video_chatgpt.constants import *

import torch
disable_torch_init()

class VideoChatGPTInterface:
    def __init__(self, 
            args_model_name, 
            args_projection_path,
            use_asr=False,
            conv_mode = "pg-video-llava",
            temperature=0.2,
            max_output_tokens=1024,
        ) -> None:
        
        
        self.use_asr=use_asr
        self.conv_mode = conv_mode
        
        model, vision_tower, tokenizer, image_processor, video_token_len = initialize_model(args_model_name, args_projection_path)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.vision_tower = vision_tower
        self.model = model
        self.temperature = temperature
        self.max_new_tokens = max_output_tokens
        self.frame_size = (image_processor.crop_size['height'], image_processor.crop_size['width'])
        
        # Create replace token, this will replace the <video> in the prompt.
        if self.model.get_model().vision_config.use_vid_start_end:
            replace_token = DEFAULT_VID_START_TOKEN + DEFAULT_VIDEO_PATCH_TOKEN * video_token_len + DEFAULT_VID_END_TOKEN
        else:
            replace_token = DEFAULT_VIDEO_PATCH_TOKEN * video_token_len
        self.replace_token = replace_token
        
        self.first_run = True
        self.state = default_conversation.copy()
        self.video_tensor_list = []
        self.video_path = None
        self.video_frames_pil = None
        self.transcript_text=None
        if self.use_asr:
            self.transcript_model = Transcriber()
        
    def clear_history(self):
        self.state = default_conversation.copy()
        self.video_tensor_list = []
        self.video_path = None
        self.video_frames_pil = None
        self.transcript_text=None
        self.first_run = True
        
    def upload_video(self, video_path):
        if isinstance(video_path, str):  # is a path
            frames = load_video(video_path, shape=self.frame_size)
            self.video_path = video_path
            self.video_frames_pil = frames
            video_tensor = self.image_processor.preprocess(frames, return_tensors='pt')['pixel_values']
            self.video_tensor_list.append(video_tensor)
            
            if self.use_asr:
                self.transcript_text = self.transcript_model.transcribe_video(video_path=video_path)
            else:
                self.transcript_text=None
        else:
            raise NotImplementedError
        
    def __get_spatio_temporal_features_torch(self, features):
        t, s, c = features.shape
        temporal_tokens = torch.mean(features, dim=1)
        padding_size = 100 - t
        if padding_size > 0:
            temporal_tokens = torch.cat((temporal_tokens, torch.zeros(padding_size, c, device=features.device)), dim=0)

        spatial_tokens = torch.mean(features, dim=0)
        concat_tokens = torch.cat([temporal_tokens, spatial_tokens], dim=0).half()

        return concat_tokens
    
    def add_text(self, text, video_path):
        if len(text) <= 0 and video_path is None:
            self.state.skip_next = True

        text = text[:1536]  # Hard cut-off
        if self.first_run:
            text = text[:1200]  # Hard cut-off for videos
            if '<video>' not in text:
                text = text + '\n<video>'
            if self.use_asr:
                text = text + '\n<audio_transcript>'
            text = (text, video_path)
            self.state = default_conversation.copy()
        self.state.append_message(self.state.roles[0], text)
        self.state.append_message(self.state.roles[1], None)
        self.state.skip_next = False
        
    
    def answer(self):
        
        if self.state.skip_next:
            return
        
        # Construct prompt
        if self.first_run:
            curr_state = self.state
            new_state = conv_templates[self.conv_mode].copy()
            new_state.append_message(new_state.roles[0], curr_state.messages[-2][1])
            new_state.append_message(new_state.roles[1], None)
            self.state = new_state
            self.first_run = False
            
        prompt =self.state.get_prompt()
        prompt = prompt.replace('<video>', self.replace_token, 1)
        prompt = prompt.replace('<audio_transcript>', f'{DEFAULT_TRANSCRIPT_START}\n\"{self.transcript_text}\"'  , 1)

        # Tokenizer
        inputs = self.tokenizer([prompt])
        input_ids = torch.as_tensor(inputs.input_ids).cuda()

        # Stopping criteria
        stop_str = self.state.sep if self.state.sep_style != SeparatorStyle.TWO else self.state.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        self.state.messages[-1][-1] = ""
        
        # Vision Encoder -  Generate video spatio-temporal features
        video_tensor = self.video_tensor_list[0]
        video_tensor = video_tensor.half().cuda()
        with torch.no_grad():
            image_forward_outs = self.vision_tower(video_tensor, output_hidden_states=True)
            select_hidden_state_layer = -2  # Same as used in LLaVA
            select_hidden_state = image_forward_outs.hidden_states[select_hidden_state_layer]
            frame_features = select_hidden_state[:, 1:]
        video_spatio_temporal_features = self.__get_spatio_temporal_features_torch(frame_features)

        # Projection and LLM
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                video_spatio_temporal_features=video_spatio_temporal_features.unsqueeze(0),
                do_sample=True,
                temperature=float(self.temperature),
                max_new_tokens=min(int(self.max_new_tokens), 1536),
                stopping_criteria=[stopping_criteria])

        # LLM output - sanity check
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            
        # Tokenizer - decoding
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        
        # Postprocessing output string
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        output = self.__post_process_code(outputs)
        for character in output:
            self.state.messages[-1][-1] += character
        
        return output
    
    def interact(self):
        print("Welcome to PG-Video-LLaVA !")
        
        video_set=False
        
        while True:
            if not video_set:
                video_path = input("Please enter the video file path:   ")
                self.upload_video(video_path)
                video_set = True
                
            try:
                text = input("USER>>")
                if not text:
                    print('----------\n\n')
                    self.clear_history()
                    video_set = False
                    continue
                    # return
                
                self.add_text(text, video_path)
                output = self.answer()
                print('ASSISTANT>>', output)
            
            except KeyboardInterrupt:
                # self.clear_history()
                print('----------')
                print('QUITTING...')
                return
    
      
    def print_state(self):
        display_txt = ""
        display_txt += "SYSTEM: " + str(self.state.system) + "\n"
        for m in self.state.messages:
            role, msg = m
            if type(msg) is tuple:
                msg, _ = msg
            display_txt += str(role) + ": " + str(msg) + "\n"
        print(display_txt)
        
    def __post_process_code(self, code):
        sep = "\n```"
        if sep in code:
            blocks = code.split(sep)
            if len(blocks) % 2 == 1:
                for i in range(1, len(blocks), 2):
                    blocks[i] = blocks[i].replace("\\_", "_")
            code = sep.join(blocks)
        return code


from grounding_evaluation.grounding_new_api import Tracker_with_GroundingDINO
from grounding_evaluation.grounding_new_api import cfg as default_cfg 
from grounding_evaluation.util.image_tagging import TaggingModule, get_unique_tags
from grounding_evaluation.util.entity_matching_openai import EntityMatchingModule
import subprocess
import glob
import tempfile
import random
import string
import datetime
import os

class PGVideoLLaVA(VideoChatGPTInterface):
    def __init__(self, args_model_name, args_projection_path, use_asr=False, conv_mode="pg-video-llava", temperature=0.2, max_output_tokens=1024) -> None:
        super().__init__(args_model_name, args_projection_path, use_asr, conv_mode, temperature, max_output_tokens)
        self.tracker = Tracker_with_GroundingDINO(
            config=default_cfg, deva_model_path=default_cfg['deva_model_path'], 
                    temporal_setting='online',
                    detection_every=5,
                    max_missed_detection_count=1,
                    max_num_objects=5, #TODO change
                    # dino_threshold=0.35
            )
        self.tagging_model = TaggingModule()
        self.entity_match_module = EntityMatchingModule()
        
    def answer(self, with_grounding=True, output_dir='outputs'):
        # Run the video-based LMM
        llm_output = super().answer()
        if not with_grounding:
            return llm_output
        
        # Apply image-tagging model
        tags_in_video = self.tagging_model.run_on_video(self.video_frames_pil)
        classes = get_unique_tags(tags_in_video)
        entity_list = classes[:10]
        # Apply entity matching model
        highlight_output, match_state = self.entity_match_module(llm_output, entity_list)
        class_list = list(set(match_state.values()))
        
        # Split the video
        print('Splitting into segments ...')
        temp_dir_splits = tempfile.TemporaryDirectory()
        temp_dir_saves = tempfile.TemporaryDirectory()
        _ = subprocess.call(["scenedetect", "-i", self.video_path, "split-video", "-o", temp_dir_splits.name])

        # For each split run tracker
        print('Running trakcer in each segment ...')
        for video_split_name in os.listdir(temp_dir_splits.name):
            # self.tracker.run_on_video(os.path.join(temp_dir_splits.name, video_split_name), os.path.join(temp_dir_saves.name,video_split_name.rsplit('.', 1)[0] + ".avi"), class_list)
            self.tracker.run_on_video(os.path.join(temp_dir_splits.name, video_split_name), os.path.join(temp_dir_saves.name,video_split_name), class_list)
            
        print('Combining output videos ...')
        # Output file path
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        _timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        _random_chars = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
        output_video_path = os.path.join(output_dir, f"video_{_timestamp}_{_random_chars}.mp4")
        output_video_path_h264 = os.path.join(output_dir, f"video_{_timestamp}_{_random_chars}_h264.mp4")
        # Combine splits and save
        mp4_files = sorted(glob.glob(os.path.join(temp_dir_saves.name, '*.mp4')))
        txt = ''
        with open(os.path.join(temp_dir_saves.name, 'video_list.txt'), 'w') as file:
            for mp4_file in mp4_files:
                txt+= "file "+ mp4_file + '\n'
            file.write(txt)
        _ = subprocess.run([f"ffmpeg -f concat -safe 0 -i {temp_dir_saves.name}/video_list.txt -c copy {output_video_path} -y"], shell=True)
        _ = subprocess.run([f"ffmpeg -i {output_video_path} -vcodec libx264 {output_video_path_h264} -y"], shell=True)
        # os.system("ffmpeg -i Video.mp4 -vcodec libx264 Video2.mp4")
        _ = subprocess.run([f"rm {temp_dir_saves.name}/video_list.txt"], shell=True)
        temp_dir_splits.cleanup()
        temp_dir_saves.cleanup()
        
        return llm_output, output_video_path_h264, highlight_output, match_state

    def interact(self):
        print("Welcome to PG-Video-LLaVA !")
        
        video_set=False
        
        while True:
            if not video_set:
                video_path = input("Please enter the video file path:   ")
                self.upload_video(video_path)
                video_set = True
                
            try:
                text = input("USER>>")
                if not text:
                    print('----------\n\n')
                    self.clear_history()
                    video_set = False
                    continue
                    # return
                
                self.add_text(text, video_path)
                llm_output, output_video_path_h264, highlight_output, match_state = self.answer(with_grounding=True)
                print('ASSISTANT>>', llm_output)
                print('\nGROUNDING>>', '\t', output_video_path_h264, '\n\t', match_state, '\n')
            
            except KeyboardInterrupt:
                # self.clear_history()
                print('----------')
                print('QUITTING...')
                return

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")

    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--projection_path", type=str, required=True)
    parser.add_argument("--use_asr", action='store_true', help='Whether to use audio transcripts or not')
    parser.add_argument("--conv_mode", type=str, required=False, default='pg-video-llava')
    parser.add_argument("--with_grounding", action='store_true', help='Run with grounding module')

    args = parser.parse_args()

    return args

if __name__=='__main__':
    args = parse_args()
    
    if args.with_grounding:
        chat = PGVideoLLaVA(
            args_model_name=args.model_name,
            args_projection_path=args.projection_path,
            use_asr=args.use_asr, 
            conv_mode=args.conv_mode,
        )
        chat.interact()
    else:
        chat = VideoChatGPTInterface(
            args_model_name=args.model_name,
            args_projection_path=args.projection_path,
            use_asr=args.use_asr, 
            conv_mode=args.conv_mode,
        )
        chat.interact()
    
    