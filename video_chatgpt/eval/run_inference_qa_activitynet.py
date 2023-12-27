import os
import argparse
import json
from tqdm import tqdm
from video_chatgpt.eval.model_utils import initialize_model, load_video
from video_chatgpt.inference import video_chatgpt_infer
from video_chatgpt.audio_transcript.transcribe import Transcriber

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument('--video_dir', help='Directory containing video files.', required=True)
    parser.add_argument('--gt_file_question', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--gt_file_answers', help='Path to the ground truth file containing answers.', required=True)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, required=False, default='pg-video-llava')
    parser.add_argument("--projection_path", type=str, required=True)
    parser.add_argument("--use_asr", action='store_true', help='Whether to use audio transcripts or not')

    return parser.parse_args()


def run_inference(args):
    """
    Run inference on ActivityNet QA DataSet using the Video-ChatGPT model.

    Args:
        args: Command-line arguments.
    """
    # Initialize the model
    model, vision_tower, tokenizer, image_processor, video_token_len = initialize_model(args.model_name,
                                                                                        args.projection_path)
    frame_size = (image_processor.crop_size['height'], image_processor.crop_size['width'])
    conv_mode = args.conv_mode
    
    # Transcript model
    if args.use_asr:
        transcript_model = Transcriber()
    
    # Load both ground truth file containing questions and answers
    with open(args.gt_file_question) as file:
        gt_questions = json.load(file)
    with open(args.gt_file_answers) as file:
        gt_answers = json.load(file)

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_list = []  # List to store the output results
    

    video_formats = ['.mp4', '.avi', '.mov', '.mkv']

    # Iterate over each sample in the ground truth file
    index = 0
    for sample in tqdm(gt_questions):
        video_name = sample['video_name']
        question = sample['question']
        id = sample['question_id']
        answer = gt_answers[index]['answer']
        index += 1

        sample_set = {'id': id, 'question': question, 'answer': answer}
        
        print('video_name:', video_name)
        
        try:
            # Load the video file
            for fmt in video_formats:  # Added this line
                temp_path = os.path.join(args.video_dir, f"v_{video_name}{fmt}")
                if os.path.exists(temp_path):
                    print('temp_path:', temp_path)
                    video_path = temp_path
                    break

            # Check if the video exists
            if os.path.exists(video_path):
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
            sample_set['pred'] = output
            output_list.append(sample_set)
            
            print('question: ', question)
            print('pred_output: ', output)
        except Exception as e:
            print(f"Error processing video file '{video_name}': {e}")

    # Save the output list to a JSON file
    with open(os.path.join(args.output_dir, f"{args.output_name}.json"), 'w') as file:
        json.dump(output_list, file)


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
