from typing import List, Optional, Tuple, Dict
import torch
import torch.nn.functional as F
import torchvision
import os
from os import path
from PIL import Image, ImagePalette
import pycocotools.mask as mask_util
from threading import Thread
from queue import Queue
from dataclasses import dataclass
import copy
import subprocess

import numpy as np
import supervision as sv

from deva.utils.pano_utils import ID2RGBConverter
from deva.inference.object_manager import ObjectManager
from deva.inference.object_info import ObjectInfo

class ImgFileWriter:
    def __init__(self, output_video_path) -> None:
        output_dir, file_extension = os.path.splitext(output_video_path)
        if not os.path.exists(output_dir): 
            os.makedirs(output_dir) 
        self.output_dir=output_dir
        self.output_video_path=output_video_path
        self.file_count=0
    
    def write(self,np_img):
        Image.fromarray(np_img).save(path.join(self.output_dir, "%09d"%self.file_count + '.jpg'))
        self.file_count+=1
        
    def release(self):
        # ffmpeg -framerate 2 -pattern_type glob -i '*.jpg' -c:v libx264 -pix_fmt yuv420p out.mp4 
        # print('LINE 1')
        _ = subprocess.run([f"ffmpeg -framerate 30 -pattern_type glob -i '{self.output_dir+'/*.jpg'}' -c:v libx264 -pix_fmt yuv420p {self.output_video_path}.mp4 "], shell=True)
        # print('LINE 2')
        _ = subprocess.run([f"rm -r {self.output_dir}"], shell=True)
        
        


class ResultSaver:
    def __init__(self,
                 output_root: str,
                 video_name: str,
                 *,
                 dataset: str,
                 object_manager: ObjectManager,
                 palette: Optional[ImagePalette.ImagePalette] = None):
        self.output_root = output_root
        self.video_name = video_name
        self.dataset = dataset.lower()
        self.palette = palette
        self.object_manager = object_manager

        
        # if self.dataset == 'gradio':
        # minimal mode, expect a cv2.VideoWriter to be assigned to self.writer asap
        self.writer = None
        self.need_remapping = True
        self.visualize = True
        self.id2rgb_converter = ID2RGBConverter()

        self.queue = Queue(maxsize=10)
        self.thread = Thread(target=save_result, args=(self.queue, ))
        self.thread.daemon = True
        self.thread.start()

    def save_mask(self,
                  prob: torch.Tensor,
                  frame_name: str,
                  need_resize: bool = False,
                  shape: Optional[Tuple[int, int]] = None,
                  save_the_mask: bool = True,
                  image_np: np.ndarray = None,
                  prompts: List[str] = None,
                  path_to_image: str = None
                  ):

        if need_resize:
            prob = F.interpolate(prob.unsqueeze(1), shape, mode='bilinear', align_corners=False)[:, 0]
        # Probability mask -> index mask
        mask = torch.argmax(prob, dim=0)

        args = ResultArgs(
            saver=self,
            mask=mask.cpu(),
            frame_name=frame_name,
            save_the_mask=save_the_mask,
            image_np=image_np,
            prompts=prompts,
            path_to_image=path_to_image,
            tmp_id_to_obj=copy.deepcopy(self.object_manager.tmp_id_to_obj),
            obj_to_tmp_id=copy.deepcopy(self.object_manager.obj_to_tmp_id),
            segments_info=copy.deepcopy(self.object_manager.get_current_segments_info()),
        )

        self.queue.put(args)

    def end(self):
        self.queue.put(None)
        self.queue.join()
        self.thread.join()


@dataclass
class ResultArgs:
    saver: ResultSaver
    mask: torch.Tensor
    frame_name: str
    save_the_mask: bool
    image_np: np.ndarray
    prompts: List[str]
    path_to_image: str
    tmp_id_to_obj: Dict[int, ObjectInfo]
    obj_to_tmp_id: Dict[ObjectInfo, int]
    segments_info: List[Dict]


def save_result(queue: Queue):
    while True:
        args: ResultArgs = queue.get()
        if args is None:
            queue.task_done()
            break

        saver = args.saver
        mask = args.mask
        frame_name = args.frame_name
        save_the_mask = args.save_the_mask
        image_np = args.image_np
        prompts = args.prompts
        path_to_image = args.path_to_image
        tmp_id_to_obj = args.tmp_id_to_obj
        obj_to_tmp_id = args.obj_to_tmp_id
        segments_info = args.segments_info
        all_obj_ids = [k.id for k in obj_to_tmp_id]

        # remap indices
        if saver.need_remapping:
            new_mask = torch.zeros_like(mask)
            for tmp_id, obj in tmp_id_to_obj.items():
                new_mask[mask == tmp_id] = obj.id
            mask = new_mask

        if saver.visualize:
            # if we are visualizing, we need to preprocess segment info
            for seg in segments_info:
                area = int((mask == seg['id']).sum())
                seg['area'] = area
            # filter out zero-area segments
            segments_info = [s for s in segments_info if s['area'] > 0]

        # save the mask to disk
        if save_the_mask:
            if saver.object_manager.use_long_id:
                out_mask = mask.numpy().astype(np.uint32)
                rgb_mask = np.zeros((*out_mask.shape[-2:], 3), dtype=np.uint8)
                for id in all_obj_ids:
                    colored_mask = saver.id2rgb_converter._id_to_rgb(id)
                    obj_mask = (out_mask == id)
                    rgb_mask[obj_mask] = colored_mask
                out_img = Image.fromarray(rgb_mask)
            else:
                out_img = Image.fromarray(mask.numpy().astype(np.uint8))
                if saver.palette is not None:
                    out_img.putpalette(saver.palette)



            if image_np is None:
                if path_to_image is not None:
                    image_np = np.array(Image.open(path_to_image))
                else:
                    raise ValueError('Cannot visualize without image_np or path_to_image')
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
                                                confidence=np.array(all_scores, dtype=np.float32),
                                                class_id=np.array(all_cat_ids))
                    
                    ### NOTE: Combined overlapping boxes of the same class.
                    
                    detections_cls_all=[]
                    for cat_id in all_cat_ids:
                        detections_cls = detections[detections.class_id == cat_id]
                        x1 = np.min(detections_cls.xyxy[:,0])
                        y1 = np.min(detections_cls.xyxy[:,1])
                        x2 = np.max(detections_cls.xyxy[:,2])
                        y2 = np.max(detections_cls.xyxy[:,3])
                        detections_cls = sv.Detections(xyxy=np.array([[x1,y1,x2,y2]], dtype=np.float32), class_id=np.array([cat_id], dtype=np.uint8))
                        detections_cls_all.append(detections_cls)
                    detections = sv.Detections.merge(detections_cls_all)
                               
                    ###
                    
                    annotator = sv.BoxAnnotator()
                    blend = annotator.annotate(scene=blend,
                                                detections=detections,
                                                labels=labels)

            
            # saver.writer.write(blend[:, :, ::-1])
            saver.writer.write(blend)

        queue.task_done()