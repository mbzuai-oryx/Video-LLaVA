import torch
from torchvision import transforms

import os
import json
from torch.utils.data import Dataset
from pathlib import Path
import time
import ffmpeg
import numpy as np
import random

from util.image_transforms import make_video_transforms, prepare


class VidSTG_Val_Test(Dataset):
    
    def __init__(self,
            vidstg_ann_path,
            vidstg_vid_path,
            image_set = 'test',
            resolution=224,
            take_only_temp_loc_frames = False,
            required_fps = 5,
            video_max_len = 100,
                
        ) -> None:
        super().__init__()
        
        self.vid_folder = vidstg_vid_path
        
        self.tmp_loc = not take_only_temp_loc_frames

        test = image_set=='test'

        if test:
            ann_file = Path(vidstg_ann_path) / f"test.json"
        elif image_set == "val":
            ann_file = Path(vidstg_ann_path) / f"val.json"
                
        self.annotations = json.load(open(ann_file, "r"))
        

        self.transforms=make_video_transforms(image_set, cautious=True, resolution=resolution)
        
        

        self.vid2imgids = ({})  # map video_id to [list of frames to be forwarded, list of frames in the annotated moment]

        for i_vid, video in enumerate(self.annotations["videos"]):
            video_fps = video["fps"]  # used for extraction
            sampling_rate = required_fps / video_fps
            assert sampling_rate <= 1  # downsampling at fps
            
            start_frame = video["start_frame"] if self.tmp_loc else video["tube_start_frame"]
            end_frame   = video["end_frame"] if self.tmp_loc else video["tube_end_frame"]
            
            frame_ids = [start_frame]
            for frame_id in range(start_frame, end_frame):
                if int(frame_ids[-1] * sampling_rate) < int(frame_id * sampling_rate):
                    frame_ids.append(frame_id)

            if len(frame_ids) > video_max_len:  # subsample at video_max_len
                frame_ids = [
                    frame_ids[(j * len(frame_ids)) // video_max_len]
                    for j in range(video_max_len)
                ]

            inter_frames = set([frame_id for frame_id in frame_ids if video["tube_start_frame"] <= frame_id < video["tube_end_frame"]]
            )  # frames in the annotated moment
            
            self.vid2imgids[video["video_id"]] = [frame_ids, inter_frames]
            # 'frame_ids' are from [start_frame:end_frame] if 'tmp_loc'==True, else [tube_start_frame:tube_end_frame]
            # 'inter_frames' are from [tube_start_frame:tube_end_frame]
            
    def __len__(self):
         return len(self.annotations["videos"])
            
    def __getitem__(self, idx):
        
        video           = self.annotations["videos"][idx]
        video_caption   = video["caption"]
        video_id        = video["video_id"] # NOTE: This 'video_id' is different from original 'vid'. This one is from 0,,,len(dataset)
        video_original_id = video["original_video_id"]
        # start_frame  = video["start_frame"]  # included #NOTE
        # end_frame  = video["end_frame"]  # excluded
        start_frame = video["start_frame"] if self.tmp_loc else video["tube_start_frame"]
        end_frame   = video["end_frame"] if self.tmp_loc else video["tube_end_frame"]
        
        frame_ids, inter_frames = self.vid2imgids[video_id] 
            # 'frame_ids' are from [start_frame:end_frame] if 'tmp_loc'==True, else [tube_start_frame:tube_end_frame]
            # 'inter_frames' are from [tube_start_frame:tube_end_frame]
            
        trajectory = self.annotations["trajectories"][video_original_id][str(video["target_id"])]
        
        # ffmpeg decoding
        vid_path = os.path.join(self.vid_folder, "video", video["video_path"])
        video_fps = video["fps"]
        ss = start_frame / video_fps
        t = (end_frame - start_frame) / video_fps
        cmd = ffmpeg.input(vid_path, ss=ss, t=t).filter("fps", fps=len(frame_ids) / t)
        out, _ = cmd.output("pipe:", format="rawvideo", pix_fmt="rgb24").run(
            capture_stdout=True, quiet=True
        )
        w = video["width"]
        h = video["height"]
        images_list = np.frombuffer(out, np.uint8).reshape([-1, h, w, 3]) # [T,H,W,3]
        assert len(images_list) == len(frame_ids)
        
        # prepare frame-level targets
        targets_list = []
        targets_list_2 = []
        inter_idx = []  # list of indexes of the frames in the annotated moment
                        # the list 'inter_idx' is the same as the 'inter_frames' that stores indices starting from idx 0.
                        
        for i_img, img_id in enumerate(frame_ids): # iterate from 'start_frame' to 'end_frame'
            if img_id in inter_frames:             # if current frame is in the annotated moment, add its annotations
                anns = trajectory[str(img_id)]  # dictionary with bbox [left, top, width, height] key
                anns = [anns]
                inter_idx.append(i_img)
            else:                               # if current frame is not in the annotated moment, ignore its annotations
                anns = []
            target = prepare(w, h, anns)
            target["image_id"] = f"{video_id}_{img_id}"
            targets_list.append(target)
            targets_list_2.append(anns)
        
        # video spatial transform
        # if self.transforms is not None:
        #     images, targets = self.transforms(images_list, targets_list)
        # else:
            # images, targets = images_list, targets_list
        images, targets = images_list, targets_list
        
        if (inter_idx):  # number of boxes should be the number of frames in annotated moment
            assert (
                len([x for x in targets if len(x["boxes"])]) == inter_idx[-1] - inter_idx[0] + 1
            ), (
                len([x for x in targets if len(x["boxes"])]), inter_idx
            )
            
        # list of bounding boxes in each frame
        img2box = {}
        for frame_id in frame_ids:
            if video["tube_start_frame"] <= frame_id < video["tube_end_frame"]:
                x1, y1, x2, y2 = self.annotations['trajectories'][video_original_id][str(video["target_id"])][str(frame_id)]["bbox"]
                img2box[frame_id] = [[x1, y1, x2, y2]]
        
        qtype = video["qtype"]
        
        
            
        # video level annotations
        video_level_ann = {
            "video_id": video_id,
            "qtype": qtype,
            "frame_ids": frame_ids,
            "inter_frames" : inter_frames,
            "inter_idx": inter_idx,
                # [inter_idx[0], inter_idx[-1]] if inter_idx
                #         else [-100,-100],  # start and end (included) indexes for the annotated moment
            "caption": video_caption,
            "img2box":img2box,
        }
        inter_idx_to_inter_frames_map = {}
        for idx, orig_frame_id in zip(video_level_ann['inter_idx'], sorted(list(video_level_ann['inter_frames']))):
            inter_idx_to_inter_frames_map[idx]=orig_frame_id
        video_level_ann["inter_idx_to_inter_frames_map"]=inter_idx_to_inter_frames_map
        
        
        
        return images, targets, qtype, video_caption, video_level_ann, targets_list_2
            # images: [3,T,H,W]
            # targets : {(['boxes', 'orig_size', 'image_id', 'size'])xT}
        

