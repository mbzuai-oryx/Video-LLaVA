import math
import whisperx
import whisper_at

######### Helper Function #####
def _slice_audio(waveform_array,sample_rate=16000, t=None, f=None):

    # frame_offset .  frame_offset+num_frames
    if t:
        t_start, t_end = t
        f_start, f_end = t_start * sample_rate, t_end * sample_rate
        f_start, f_end = math.floor(f_start), math.floor(f_end)
    elif f:
        f_start,f_end = f
        
    new_waveform_array = waveform_array[f_start:f_end]
    
    return new_waveform_array


######### Transcriber #########
class Transcriber: 
    def __init__(self,
            device = "cuda", #The device to load the model on.
            device_index = 0, #gpu_id
            batch_size = 16, # reduce if low on GPU mem
            compute_type = "float16", # change to "int8" if low on GPU mem (may reduce accuracy)
            whisper_variant = "base", # whisper_variant = "large-v2"
        ) -> None:
        ''' 
            device : The device to load the model ['cpu'|'gpu']
            device_index: GPU ID
            batch_size: batch_size for whisperX (reduce this if low on GPU memory)
            compute_type: default is "float16". change to "int8" if low on GPU memory (may reduce accuracy)
            whisper_variant:  ["base", "large-v2"]
        '''
        
        self.device = device
        self.batch_size = batch_size
        self.compute_type = compute_type
        self.whisper_variant = whisper_variant
        
        self.whisperx_model = whisperx.load_model(whisper_variant, device, device_index=device_index, compute_type=compute_type)
        self.alignment_model_en, self.alignment_metadata = whisperx.load_align_model(language_code="en", device=device);
        
        device_wat = device
        if device=='cuda':
            device_wat=device+':'+str(device_index)
        self.whisper_at_model = whisper_at.load_model(whisper_variant, device=device_wat)

    def transcribe_video(self, video_path, return_text_only = True):
        
        audio_array = whisperx.load_audio(video_path)
        
        # 1. Generate transcription from whisperx
        whisper_original_result = self.whisperx_model.transcribe(audio_array, batch_size=self.batch_size);
        
        all_segments = []

        if whisper_original_result["language"]=='en':
            # 2. Align whisper output using a Phoneme model
            whisperx_alignment_result = whisperx.align(whisper_original_result["segments"], self.alignment_model_en, self.alignment_metadata, audio_array, self.device, return_char_alignments=False);
            
            # 3. Filter each transcript segment using the audio-tagging output
            for segment in whisperx_alignment_result['segments']:
            
                t_start, t_end, sentence = segment['start'], segment['end'], segment['text']
                
                subclip = _slice_audio(audio_array, t=(t_start,t_end))
                at_time_res_approx = round(int(math.ceil((t_end-t_start)/0.4)),1) *40/100  
                at_result = self.whisper_at_model.transcribe(subclip, at_time_res= at_time_res_approx)
                
                audio_tag_result = whisper_at.parse_at_label(at_result, language='en', top_k=3, p_threshold=-5, include_class_list=list(range(527)))[0]

                audio_tag_names = [k[0] for k in audio_tag_result['audio tags'] ]
                if not ("Speech" in audio_tag_names or "Male speech, man speaking" in audio_tag_names or "Female speech, woman speaking" in audio_tag_names):
                    continue
                if "Music" in audio_tag_names:
                    music_prob = [k[1] for k in audio_tag_result['audio tags'] if k[0]=="Music" ][0]
                    speech_prob = [k[1] for k in audio_tag_result['audio tags'] if k[0]=="Speech" or k[0]=="Male speech, man speaking" or k[0]=="Female speech, woman speaking"  ][0]
                    if music_prob>speech_prob and abs(music_prob-speech_prob)>1.1:
                        continue
                all_segments.append({"text":sentence, 
                                    "time":{"start":t_start, "end": t_end},
                                    "audio_tags": audio_tag_result['audio tags']
                                    })
            
        else:
            print("Non-English language: ", whisper_original_result["language"])
        
        # return the text only
        if return_text_only:
            transcript = ''
            for seg in all_segments:
                transcript += seg['text']
            transcript = transcript.strip()
            return transcript
        
        return all_segments

