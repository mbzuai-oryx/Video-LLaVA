import os
import json
from torch.utils.data import Dataset
from pathlib import Path
from util.image_transforms import make_video_transforms, prepare
import time
import ffmpeg
import numpy as np
import random
from PIL import Image
import copy

class HCSTVG_Dataset(Dataset):
    def __init__(
        self,
        vid_folder,
        ann_file,
        image_set = 'test',
        take_only_temp_loc_frames = False,
        video_max_len=100,
        required_fps=5,
        resolution=224,
    ):
        """
        :param vid_folder: path to the folder containing a folder "video"
        :param ann_file: path to the json annotation file
        :param transforms: video data transforms to be applied on the videos and boxes
        :param is_train: whether training or not
        :param video_max_len: maximum number of frames to be extracted from a video
        :param video_max_len_train: maximum number of frames to be extracted from a video at training time
        :param fps: number of frames per second
        :param tmp_loc: whether to use temporal localization annotations
        """
        self.vid_folder = vid_folder
        print("loading annotations into memory...")
        tic = time.time()
        self.annotations = json.load(open(ann_file, "r"))
        print("Done (t={:0.2f}s)".format(time.time() - tic))
        self._transforms =make_video_transforms(image_set, cautious=True, resolution=resolution),
        self.video_max_len = video_max_len
        self.tmp_loc = not take_only_temp_loc_frames
        
        self.vid2imgids = {}
        for i_vid, video in enumerate(self.annotations):
            video_num_images = video["frame_count"]
            video_fps = video_num_images / 20  # duration of videos in HC-STVG is 20s
            sampling_rate = required_fps / video_fps
            assert sampling_rate <= 1  # downsampling at fps
            start_frame = 0 if self.tmp_loc else video["tube_start_frame"]
            end_frame = (
                video_num_images - 1 if self.tmp_loc else video["tube_end_frame"]
            )
            frame_ids = [start_frame]
            for frame_id in range(start_frame, end_frame):
                if int(frame_ids[-1] * sampling_rate) < int(frame_id * sampling_rate):
                    frame_ids.append(frame_id)

            if len(frame_ids) > video_max_len:  # subsample at video_max_len
                frame_ids = [
                    frame_ids[(j * len(frame_ids)) // video_max_len]
                    for j in range(video_max_len)
                ]

            # inter_frames = set([frame_id for frame_id in frame_ids 
            #         if video["tube_start_frame"] <= frame_id < video["tube_end_frame"]
            #     ])
            inter_frames = []
            for frame_id in frame_ids:
                if video["tube_start_frame"] <= frame_id < video["tube_end_frame"]:
                    # x1, y1, w, h = video["trajectory"][frame_id - video["tube_start_frame"]]
                    # x2 = x1 + w
                    # y2 = y1 + h
                    # self.img2box[f"{video_id}_{frame_id}"] = [[x1, y1, x2, y2]]
                    inter_frames.append(frame_id)
            
            self.vid2imgids[video["video_id"]] = [frame_ids, inter_frames]

    def __len__(self) -> int:
        return len(self.annotations)
    
    def get_caption(self, idx):
        video = self.annotations[idx]
        video_caption = video["caption"]
        return video_caption

    def __getitem__(self, idx):
        """
        :param idx: int
        :return:
        images: a CTHW video tensor
        targets: list of frame-level target, one per frame, dictionary with keys image_id, boxes, orig_sizes
        tmp_target: video-level target, dictionary with keys video_id, inter_idx, frames_id, caption
        """
        video = self.annotations[idx]
        video_caption = video["caption"]
        video_id = video["video_id"]
        video_original_id = video["original_video_id"]
        trajectory = video["trajectory"]
        frame_ids, inter_frames = self.vid2imgids[video_id]

        video_num_images = video["frame_count"]
        video_fps = video_num_images / 20
        
        start_frame = 0 if self.tmp_loc else video["tube_start_frame"]
        end_frame   = video_num_images - 1 if self.tmp_loc else video["tube_end_frame"]
            
        # ffmpeg decoding
        # vid_path = os.path.join(self.vid_folder, "video", video["video_path"])
        vid_path = os.path.join(self.vid_folder, video["video_path"])
        # ss = 0
        # t = 20
        ss = start_frame / video_fps
        t = (end_frame - start_frame) / video_fps
        # print('vid_path: ', vid_path)
        cmd = ffmpeg.input(vid_path, ss=ss, t=t).filter("fps", fps=len(frame_ids) / t)
        out, _ = cmd.output("pipe:", format="rawvideo", pix_fmt="rgb24").run(
            capture_stdout=True, quiet=True
        )
        w = video["width"]
        h = video["height"]
        images_list = np.frombuffer(out, np.uint8).reshape([-1, h, w, 3])
        assert len(images_list) == len(frame_ids)

        # prepare frame-level targets
        targets_list = []
        targets_list_2 = []
        inter_idx = []  # list of indexes of the frames in the annotated moment
        for i_img, img_id in enumerate(frame_ids):
            if img_id in inter_frames:
                bbox = trajectory[img_id - video["tube_start_frame"]]  # dictionary with bbox [left, top, width, height] key
                # bbox = trajectory[i_img] # dictionary with bbox [left, top, width, height] key #NOTE
                anns = {"bbox": bbox}
                anns = [anns]

                anns2 = {"bbox": [bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3]]}
                anns2 = [anns2]
                inter_idx.append(i_img)
            else:
                anns = []
                anns2 = []
            target = prepare(w, h, anns)
            target["image_id"] = f"{video_id}_{img_id}"
            targets_list.append(target)
            targets_list_2.append(anns2)

        # video spatial transform
        # if self._transforms is not None:
        #     images, targets = self._transforms(images_list, targets_list)
        # else:
        #     images, targets = images_list, targets_list
        images, targets = images_list, targets_list
        
        # print('frame_ids:', frame_ids)
        # print('targets:', targets)

        # if inter_idx:
        #     assert ( len([x for x in targets if len(x["boxes"])]) == inter_idx[-1] - inter_idx[0] + 1
        #         ),f"{ len([x for x in targets if len(x['boxes'])]) },  \n inter_idx: {inter_idx }" # , len([x for x in bis if len(x["boxes"])])


        #######
        img2box = {} 
        for frame_id in frame_ids:
            if video["tube_start_frame"] <= frame_id < video["tube_end_frame"]:
                x1, y1, w, h = video["trajectory"][frame_id - video["tube_start_frame"]] 
                x2 = x1 + w
                y2 = y1 + h
                img2box[frame_id] = [[x1, y1, x2, y2]]
        #######


        qtype = 'declarative'
        
        # video level annotations
        video_level_ann = {
            "video_id": video_id,
            "qtype": qtype,
            "frame_ids": frame_ids,
            "inter_frames" : inter_frames,
            # "inter_idx": [inter_idx[0], inter_idx[-1]] if inter_idx else [-100, -100],
            "inter_idx": inter_idx,
            "caption": video_caption,
            "img2box":img2box,
        }
        inter_idx_to_inter_frames_map = {}
        for idx, orig_frame_id in zip(video_level_ann['inter_idx'], sorted(list(video_level_ann['inter_frames']))):
            inter_idx_to_inter_frames_map[idx]=orig_frame_id
        video_level_ann["inter_idx_to_inter_frames_map"]=inter_idx_to_inter_frames_map

        return images, targets, qtype, video_caption, video_level_ann, targets_list_2


