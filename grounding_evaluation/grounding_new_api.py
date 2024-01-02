import torchvision

import os
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import copy

import sys

import os
from os import path
from argparse import ArgumentParser

import torch
import numpy as np

from deva.model.network import DEVA
from deva.inference.inference_core import DEVAInferenceCore
# from deva.inference.result_utils import ResultSaver
from grounding_evaluation.util.result_utils import ResultSaver, ImgFileWriter
from deva.inference.demo_utils import flush_buffer
from deva.ext.grounding_dino import get_grounding_dino_model

from tqdm import tqdm
import json

torch.autograd.set_grad_enabled(False)

# for id2rgb
np.random.seed(42)


#### Args ################

saved_chekpoint_dir = "grounding_evaluation/weights"

args_original=argparse.Namespace(
    deva_model_path = f'{saved_chekpoint_dir}/DEVA-propagation.pth',
    amp = False,
    
    # Model parameters
    key_dim=64,
    value_dim=512,
    pix_feat_dim=512,
    
    # Long-term memory options
    disable_long_term=False,
    max_mid_term_frames=10, # 'T_max in XMem, decrease to save memory'
    min_mid_term_frames=5, # 'T_min in XMem, decrease to save memory'
    max_long_term_elements=10000, # 'LT_max in XMem, increase if objects disappear for a long time'
    num_prototypes=128, #'P in XMem'
    top_k=30,
    mem_every=5, #'r in XMem. Increase to improve running speed.'
    chunk_size=1,#Number of objects to process in parallel as a batch; -1 for unlimited. Set to a small number to save memory.
    size=480, #Resize the shorter side to this size. -1 to use original resolution. 
    
    
     # Grounded Segment Anything
    GROUNDING_DINO_CONFIG_PATH = f'{saved_chekpoint_dir}/GroundingDINO_SwinT_OGC.py',
    GROUNDING_DINO_CHECKPOINT_PATH= f'{saved_chekpoint_dir}/groundingdino_swint_ogc.pth',
    DINO_THRESHOLD = 0.35,
    DINO_NMS_THRESHOLD=0.8,
    
    # Segment Anything (SAM) models
    SAM_ENCODER_VERSION='vit_h',
    SAM_CHECKPOINT_PATH=f'{saved_chekpoint_dir}/sam_vit_h_4b8939.pth',
    # Mobile SAM
    MOBILE_SAM_CHECKPOINT_PATH=f'{saved_chekpoint_dir}/mobile_sam.pt',
    
    # Segment Anything (SAM) parameters
    SAM_NUM_POINTS_PER_SIDE=64, #'Number of points per side for prompting SAM'
    SAM_NUM_POINTS_PER_BATCH=64, #'Number of points computed per batch',
    SAM_PRED_IOU_THRESHOLD=0.88, #'(Predicted) IoU threshold for SAM'
    SAM_OVERLAP_THRESHOLD=0.8, #'Overlap threshold for overlapped mask suppression in SAM'
    
    # Default Args for tracking with text prompt
    detection_every=5,
    num_voting_frames=3, #'Number of frames selected for voting. only valid in semionline'
    temporal_setting='semionline', #semionline/online
    max_missed_detection_count=10,
    max_num_objects=-1, #'Max. num of objects to keep in memory. -1 for no limit'
    sam_variant='original' #'mobile/original'
    
)

##########################

cfg = vars(args_original)
cfg['enable_long_term'] = not cfg['disable_long_term'] # 'True'

cfg['enable_long_term_count_usage'] = True
cfg['max_num_objects'] = 50
cfg['size'] = 480
# cfg['DINO_THRESHOLD'] = 0.35
cfg['amp'] = True
cfg['chunk_size'] = 4
# cfg['detection_every'] = 5
cfg['detection_every'] = 5 #2 # NOTE: this was originally 5
cfg['max_missed_detection_count'] = 5 #NOTE: this was originally 10
# cfg['sam_variant'] = 'original'
cfg['sam_variant'] =  'mobile' #'original' #NOTE: 
cfg['temporal_setting'] = 'semionline' #'semionline' # NOTE: this was originally 'online' # semionline usually works better; but online is faster for this demo
cfg['pluralize'] = True
# 
cfg['DINO_THRESHOLD'] = 0.5

# CLIP model
import clip

imagenet_templates = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]

def mask_and_crop(image: np.ndarray, xyxy: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Crops the given image based on the given bounding box.
    Args:
        image (np.ndarray): The image to be cropped, represented as a numpy array.
        xyxy (np.ndarray): A numpy array containing the bounding box coordinates in the format (x1, y1, x2, y2).
    Returns:
        (np.ndarray): The cropped image as a numpy array.
    Examples:
        ```python
        >>> import supervision as sv
        >>> detection = sv.Detections(...)
        >>> with sv.ImageSink(target_dir_path='target/directory/path') as sink:
        ...     for xyxy in detection.xyxy:
        ...         cropped_image = sv.crop(image=image, xyxy=xyxy)
        ...         sink.save_image(image=image)
        ```
    """

    # xyxy = np.round(xyxy).astype(int)
    # x1, y1, x2, y2 = xyxy
    
    new_image = image.copy()
    new_image[mask==0]=0
    
    #
    x1,y1,x2,y2 = xyxy
    x1,y1,x2,y2 = np.floor(x1),np.floor(y1), np.ceil(x2),np.ceil(y2)
    x1,y1,x2,y2 = x1.astype('int'), y1.astype('int'), x2.astype('int'), y2.astype('int')
    x1=max(0, x1)
    y1=max(0, y1)
    y2=min(y2, image.shape[0])
    x2=min(x2, image.shape[1])
    cropped_img = new_image[y1:y2, x1:x2]
    return cropped_img


from deva.inference.demo_utils import get_input_frame_for_deva
from deva.inference.frame_utils import FrameInfo
from deva.inference.object_info import ObjectInfo
from typing import Dict, List
import torch.nn.functional as F

import cv2


class Tracker_with_GroundingDINO:
    def __init__(self, config,
                 deva_model_path,
                 device='cuda',
                 temporal_setting=None,
                 detection_every = None,
                 max_missed_detection_count = None,
                 max_num_objects = None,
                 dino_threshold=None,
                 ) -> None:
        self.use_amp = config['amp']
        self.tracker_cfg = {
            'size': config['size'],
            'temporal_setting' : temporal_setting if temporal_setting else config['temporal_setting'],
            'num_voting_frames': config['num_voting_frames'],
            'detection_every'  : detection_every if detection_every else config['detection_every'],
            'value_dim' :config['value_dim'],
            'chunk_size':config['chunk_size'],
            'top_k':config['top_k'],
            'mem_every':config['mem_every'],
            'enable_long_term':config['enable_long_term'],
            'chunk_size':config['chunk_size'],
            'max_missed_detection_count': max_missed_detection_count if max_missed_detection_count else config.get('max_missed_detection_count'),
            'max_num_objects':max_num_objects if max_num_objects else config.get('max_num_objects'),
            'enable_long_term':config['enable_long_term'],
            'enable_long_term_count_usage':config['enable_long_term_count_usage'],
            'enable_long_term_count_usage':config['enable_long_term_count_usage'],
            'max_mid_term_frames': config['max_mid_term_frames'],
            'min_mid_term_frames': config['min_mid_term_frames'],
            'num_prototypes': config['num_prototypes'],
            'max_long_term_elements' :config['max_long_term_elements'],
        }
        self.dino_config = {
            'DINO_THRESHOLD':dino_threshold if dino_threshold else config['DINO_THRESHOLD'],
            'DINO_NMS_THRESHOLD':config['DINO_NMS_THRESHOLD']
        }
        
        # Load DEVA checkpoint
        deva_model_cfg = {
            'key_dim':config['key_dim'],
            'value_dim':config['value_dim'],
            'pix_feat_dim':config['pix_feat_dim'],
        }
        self.deva_model = DEVA(deva_model_cfg).cuda().eval()
        self.deva_model.load_weights(torch.load(deva_model_path))
        
        gd_model, sam_model = get_grounding_dino_model(config, 'cuda')
        self.sam = sam_model
        self.gd_model = gd_model
        
        self.deva=None
        
        self.clip_device = device
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
        
        self.tracking_results = []


    def __get_clip_zeroshot_classifier(self, classnames, templates):
        with torch.no_grad():
            zeroshot_weights = []
            for classname in classnames:
                texts = [template.format(classname) for template in templates] #format with class
                texts = clip.tokenize(texts).cuda() #tokenize
                class_embeddings = self.clip_model.encode_text(texts) #embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(self.clip_device)
        return zeroshot_weights

    def __classify(self, pre_processed_imgs, zeroshot_weights):
        with torch.no_grad():
        
            images = pre_processed_imgs.to(self.clip_device)
            
            # predict
            image_features = self.clip_model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            # logits_per_image = 100. * image_features @ zeroshot_weights
            clip_scores = image_features @ zeroshot_weights
            logits_per_image = 100. * clip_scores
            
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            
            clip_scores = clip_scores.cpu().numpy()
            
            return clip_scores, probs
        
    def __apply_clip_thresholding(self, image_rgb, detections, class_list, clip_zeroshot_weights):
        # Crop detected boxes
        crops = [mask_and_crop(image=image_rgb, xyxy=detections[i].xyxy[0], mask=detections[i].mask[0]) for i in range(len(detections))]
        if crops:
            cropped_images = [Image.fromarray( cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) ) for crop in crops]
            pre_processed_imgs =  torch.stack([self.clip_preprocess(cropped_image) for cropped_image in cropped_images]).to(self.clip_device)
            # classify crops with CLIP
            clip_scores, probs = self.__classify(pre_processed_imgs, clip_zeroshot_weights)
            # pred_classes = [class_list[np.argmax(prob)] if prob.max() > 0.1 else None for prob in probs]
            pred_classes = [class_list[np.argmax(probs[i])] if probs[i].max() > 0.1 and clip_scores[i][np.argmax(probs[i])]>=0.2 else None for i in range(len(probs))]

            # Threshold based on CLIP score
            # print(f"Before CLIP-score based thresholding: {len(detections.xyxy)} boxes")
            thresholded_idx = [i for i in range(len(pred_classes)) if pred_classes[i] == class_list[detections.class_id[i]]]
            detections.xyxy = detections.xyxy[thresholded_idx]
            detections.confidence = detections.confidence[thresholded_idx]
            detections.class_id = detections.class_id[thresholded_idx]
            detections.mask = detections.mask[thresholded_idx]
            # print(f"After CLIP-score based thresholding: {len(detections.xyxy)} boxes")
            
            return detections
        else:
            return detections


    def init_tracker(self, class_list, output_video_path=None):
        self.tracking_results = []
        
        self.deva = DEVAInferenceCore(self.deva_model, config=self.tracker_cfg)
        self.deva.next_voting_frame = self.tracker_cfg['num_voting_frames'] - 1
        self.deva.enabled_long_id()
        self.CLASSES = class_list
        self.clip_zeroshot_weights = self.__get_clip_zeroshot_classifier(class_list, imagenet_templates)
        
        self.output_video_path=output_video_path            
        if self.output_video_path:
            self.result_saver = ResultSaver(None, None, dataset='gradio', object_manager=self.deva.object_manager)
            self.result_writer_initialized = False
        
        self.ti=0

    def detect_and_segment_on_frame(self,
                        image: np.ndarray, 
                        prompts: List[str],
                        min_side: int) -> (torch.Tensor, List[ObjectInfo]):
        """
        config: the global configuration dictionary
        image: the image to segment; should be a numpy array; H*W*3; unnormalized (0~255)
        prompts: list of class names

        Returns: a torch index mask of the same size as image; H*W
                a list of segment info, see object_utils.py for definition
        """

        BOX_THRESHOLD = TEXT_THRESHOLD = self.dino_config['DINO_THRESHOLD']
        NMS_THRESHOLD = self.dino_config['DINO_NMS_THRESHOLD']

        self.sam.set_image(image, image_format='RGB') #NOTE: 'image' is RGB

        # detect objects
        # GroundingDINO uses BGR
        detections = self.gd_model.predict_with_classes(image=cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                                                classes=prompts,
                                                box_threshold=BOX_THRESHOLD,
                                                text_threshold=TEXT_THRESHOLD)

        # NMS post processing
        nms_idx = torchvision.ops.nms(torch.from_numpy(detections.xyxy),
                                    torch.from_numpy(detections.confidence),
                                    NMS_THRESHOLD).numpy().tolist()

        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]

        # Get segmentation masks from SAM
        result_masks = []
        for box in detections.xyxy:
            masks, scores, _ = self.sam.predict(box=box, multimask_output=True)
            index = np.argmax(scores)
            result_masks.append(masks[index])
        detections.mask = np.array(result_masks)
        
        
        # Apply Clip based filtering
        detections = self.__apply_clip_thresholding(image, detections, self.CLASSES, self.clip_zeroshot_weights)

        #
        h, w = image.shape[:2]
        if min_side > 0:
            scale = min_side / min(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
        else:
            new_h, new_w = h, w

        output_mask = torch.zeros((new_h, new_w), dtype=torch.int64, device=self.gd_model.device)
        curr_id = 1
        segments_info = []

        # sort by descending area to preserve the smallest object
        for i in np.flip(np.argsort(detections.area)):
            mask = detections.mask[i]
            confidence = detections.confidence[i]
            class_id = detections.class_id[i]
            mask = torch.from_numpy(mask.astype(np.float32))
            mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), (new_h, new_w), mode='bilinear')[0, 0]
            mask = (mask > 0.5).float()

            if mask.sum() > 0:
                output_mask[mask > 0] = curr_id
                segments_info.append(ObjectInfo(id=curr_id, category_id=class_id, score=confidence))
                curr_id += 1

        return output_mask, segments_info

        
    @torch.inference_mode()
    def process_frame_fn(self,
                                ti: int,
                                image_np: np.ndarray = None) -> None:
        '''
            Inputs:
                - ti        :
                - image_np  :   RGB image as numpy array
        '''
        # frame_path = 'null.png' 
        # image_np, if given, should be in RGB
        if image_np is None:
            # image_np = cv2.imread(frame_path)
            # image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            raise Exception('"image_np" cannot be None')
        
        prompts = self.CLASSES

        h, w = image_np.shape[:2]
        new_min_side = self.tracker_cfg['size']
        need_resize = new_min_side > 0
        image = get_input_frame_for_deva(image_np, new_min_side)

        # frame_name = path.basename(frame_path)
        frame_name = 'null.png'
        frame_info = FrameInfo(image, None, None, ti, {
            'frame': [frame_name],
            'shape': [h, w],
        })

        if self.tracker_cfg['temporal_setting'] == 'semionline':
            if ti + self.tracker_cfg['num_voting_frames'] > self.deva.next_voting_frame:
                mask, segments_info = self.detect_and_segment_on_frame(image_np, prompts, new_min_side)
                
                frame_info.mask = mask
                frame_info.segments_info = segments_info
                frame_info.image_np = image_np  # for visualization only
                # wait for more frames before proceeding
                self.deva.add_to_temporary_buffer(frame_info)

                if ti == self.deva.next_voting_frame:
                    # process this clip
                    this_image = self.deva.frame_buffer[0].image
                    this_frame_name = self.deva.frame_buffer[0].name
                    this_image_np = self.deva.frame_buffer[0].image_np

                    _, mask, new_segments_info = self.deva.vote_in_temporary_buffer(keyframe_selection='first')
                    prob = self.deva.incorporate_detection(this_image, mask, new_segments_info)
                    self.deva.next_voting_frame += self.tracker_cfg['detection_every']
                    self.save_result_on_frame(prob, this_frame_name, need_resize=need_resize, shape=(h, w), image_np=this_image_np, prompts=prompts)

                    for frame_info in self.deva.frame_buffer[1:]:
                        this_image = frame_info.image
                        this_frame_name = frame_info.name
                        this_image_np = frame_info.image_np
                        prob = self.deva.step(this_image, None, None)
                        self.save_result_on_frame(prob, this_frame_name, need_resize=need_resize, shape=(h, w), image_np=this_image_np, prompts=prompts)

                    self.deva.clear_buffer()
            else:
                # standard propagation
                prob = self.deva.step(image, None, None)
                self.save_result_on_frame(prob, frame_name, need_resize=need_resize, shape=(h, w), image_np=image_np, prompts=prompts)

        elif self.tracker_cfg['temporal_setting'] == 'online':
            if ti % self.tracker_cfg['detection_every'] == 0:
                # incorporate new detections
                mask, segments_info = self.detect_and_segment_on_frame(image_np, prompts, new_min_side)
                frame_info.segments_info = segments_info
                prob = self.deva.incorporate_detection(image, mask, segments_info)
            else:
                # Run the model on this frame
                prob = self.deva.step(image, None, None)
            self.save_result_on_frame(prob, frame_name, need_resize=need_resize, shape=(h, w), image_np=image_np, prompts=prompts)

        
    def run_on_frame(self, frame, save_at_fps=5):
        '''
            Inputs:
                - frame    : RGB image as numpy array
                - save_at_fps : 
        '''
        if self.output_video_path:
            if not self.result_writer_initialized:
                h, w = frame.shape[:2]
                self.result_writer_initialized = True
                # self.result_saver.writer = cv2.VideoWriter(self.output_video_path, cv2.VideoWriter_fourcc(*'MJPG'), save_at_fps, (w, h))
                self.result_saver.writer = ImgFileWriter(self.output_video_path)
            
        self.process_frame_fn(self.ti, image_np=frame)
        self.ti += 1
        
    def clear_tracker(self):
        if self.output_video_path:
            flush_buffer(self.deva, self.result_saver)
            self.result_saver.writer.release()
        self.deva.clear_buffer()
        del self.deva
        
        self.result_saver = None
        self.result_writer_initialized = False
        
    def save_result_on_frame(self, prob, frame_name, need_resize, shape, image_np, prompts):
        if self.output_video_path:
            self.result_saver.save_mask(prob, frame_name, need_resize=need_resize, shape=shape, image_np=cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR), prompts=prompts)

        if need_resize:
            prob = F.interpolate(prob.unsqueeze(1), shape, mode='bilinear', align_corners=False)[:, 0]
        mask = torch.argmax(prob, dim=0)
        mask = mask.cpu()
        tmp_id_to_obj=copy.deepcopy(self.deva.object_manager.tmp_id_to_obj)
        obj_to_tmp_id=copy.deepcopy(self.deva.object_manager.obj_to_tmp_id)
        segments_info=copy.deepcopy(self.deva.object_manager.get_current_segments_info())
        all_obj_ids = [k.id for k in obj_to_tmp_id]
        
        self.tracking_results.append({
            'mask': mask,
            'tmp_id_to_obj':tmp_id_to_obj,
            'obj_to_tmp_id':obj_to_tmp_id,
            'segments_info':segments_info,
            'all_obj_ids':all_obj_ids,
            'image_np':image_np,
            'prompts':prompts,
        })
        

    def run_on_video(self, input_video_path, output_video_path, class_list):
        self.init_tracker(class_list, output_video_path)

        cap = cv2.VideoCapture(input_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))) as pbar:
                while (cap.isOpened()):
                    ret, frame = cap.read()
                    if ret == True:
                        self.run_on_frame(frame, save_at_fps=fps)
                        pbar.update(1)
                    else:
                        break
        cap.release()
        self.clear_tracker()
        
    def run_on_list_of_images(self, img_list, class_list):
        self.init_tracker(class_list, output_video_path=None)
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            for img in img_list:
                self.run_on_frame(np.array(img))
        self.clear_tracker()
        return self.tracking_results
    



if __name__=='__main__':
    
    parser = ArgumentParser()
    parser.add_argument(
        "--INPUT_PATH", type=str, required=True,
        help="input video'",
    )
    parser.add_argument(
        "--OUTPUT_PATH", type=str, default="./out_new.mp4", help="the video path"
    )
    parser.add_argument(
        "--CAPTION", required=True, help="text prompt for GroundingDINO",
        nargs='+',
    )
    args = parser.parse_args()
            
    tracker = Tracker_with_GroundingDINO(config=cfg, deva_model_path=cfg['deva_model_path'], device='cuda')
    
    tracker.run_on_video(args.INPUT_PATH, args.OUTPUT_PATH, args.CAPTION)
    