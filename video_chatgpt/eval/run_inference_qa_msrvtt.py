# videochat
import os
import argparse
import json
from tqdm import tqdm
from video_chatgpt.eval.model_utils import initialize_model, load_video
from video_chatgpt.inference import video_chatgpt_infer
from video_chatgpt.audio_transcript.transcribe import Transcriber



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', help='dir containing video files', required=True)
    parser.add_argument('--gt_file', help='path to gt', required=True)
    parser.add_argument('--output_dir', help='dir to save model result json', required=True)
    parser.add_argument('--output_name', help='name of the file for storing result json', required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, required=False, default='pg-video-llava')
    parser.add_argument("--projection_path", type=str, required=True)
    parser.add_argument("--use_asr", action='store_true', help='Whether to use audio transcripts or not')
    args = parser.parse_args()

    return args

def run_inference(args):
    # Initialize the model
    model, vision_tower, tokenizer, image_processor, video_token_len = initialize_model(args.model_name,
                                                                                        args.projection_path)
    frame_size = (image_processor.crop_size['height'], image_processor.crop_size['width'])
    conv_mode = args.conv_mode
    
    # Transcript model
    if args.use_asr:
        transcript_model = Transcriber()
        
    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    # process the query and pass the video. Iterate through each sample in validation
    # load gt file
    file = open(args.gt_file)
    gt_contents = json.load(file)

    output_list = []

    for sample in tqdm(gt_contents):
        video_id = sample['video_id']
        question = sample['question']
        video_name = f"video{sample['video_id']}"
        output_set = sample
        
        try:


            video_path = os.path.join(args.video_dir, f"{video_name}.mp4")
            video_frames = load_video(video_path, shape=frame_size)
            
            if args.use_asr:
                try:
                    transcript_text = transcript_model.transcribe_video(video_path=video_path)
                except:
                    transcript_text = None
            else:
                transcript_text=None
            
            # Run inference on the video and add the output to the list
            output = video_chatgpt_infer(video_frames, question, conv_mode, model, vision_tower,
                                            tokenizer, image_processor, video_token_len, transcript_text)
            output_set['pred'] = output
            output_list.append(output_set)
        except Exception as e:
            print(f"Error processing video file '{video_name}': {e}")

    # Save the output_list to a JSON file
    with open(os.path.join(args.output_dir, f"{args.output_name}.json"), 'w') as file:
        json.dump(output_list, file)


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
