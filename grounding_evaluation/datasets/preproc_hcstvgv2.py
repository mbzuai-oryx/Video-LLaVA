import json
import os
from tqdm import tqdm
import argparse

if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description="Preprocess HC-STVG Dataset")
    parser.add_argument("--video_dir", required=True, help="The path to the directory containing HC-STVG videos")
    parser.add_argument("--ann_dir", required=True, help="The path to the directory containing HC-STVG annotations")
    args = parser.parse_args()
    
    # video_path = "/share/users/shehan/data/HC-STVG/Video"
    # ann_path = "/share/users/shehan/data/HC-STVG/anno_v2"
    
    video_path = args.video_dir
    ann_path = args.ann_dir

    # get video to path mapping
    dirs = os.listdir(video_path)
    vid2path = {}
    for dir in dirs:
        files = os.listdir(os.path.join(video_path, dir))
        for file in files:
            assert os.path.exists(os.path.join(video_path, dir, file))
            vid2path[file[:-4]] = os.path.join(dir, file)

    # preproc annotations
    files = ["train_v2.json", "val_v2.json"]
    for file in files:
        videos = []
        annotations = json.load(open(os.path.join(ann_path, file), "r"))
        for video, annot in tqdm(annotations.items()):
            out = {
                "original_video_id": video[:-4],
                "frame_count": annot["img_num"],
                "width": annot["img_size"][1],
                "height": annot["img_size"][0],
                "tube_start_frame": annot["st_frame"],  # starts with 1
                "tube_end_frame": annot["st_frame"] + len(annot["bbox"]),  # excluded
                "tube_start_time": annot["st_time"],
                "tube_end_time": annot["ed_time"],
                "video_path": vid2path[video[:-4]],
                "caption": annot["English"],
                "video_id": len(videos),
                "trajectory": annot["bbox"],
            }
            videos.append(out)

        json.dump(videos, open(os.path.join(ann_path, file[:-5] + "_proc.json"), "w"))