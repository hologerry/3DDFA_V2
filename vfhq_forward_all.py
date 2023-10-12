import json
import os

from argparse import ArgumentParser

import cv2
import numpy as np
import torch.backends.cudnn as cudnn
import yaml

from tqdm import tqdm

from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.pose import calc_pose


def crop_square(img, size=256, interpolation=cv2.INTER_AREA):
    h, w = img.shape[:2]
    min_size = np.amin([h, w])

    # Centralize and crop
    crop_img = img[
        int(h / 2 - min_size / 2) : int(h / 2 + min_size / 2),
        int(w / 2 - min_size / 2) : int(w / 2 + min_size / 2),
    ]
    resized = cv2.resize(crop_img, (size, size), interpolation=interpolation)

    return resized


def main(args):
    assert args.job_idx < args.job_num
    assert args.gpu_idx < args.gpu_num

    cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)
    cudnn.benchmark = True
    gpu_mode = True
    tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)
    face_boxes = FaceBoxes()

    splits = ["VFHQ-Train", "VFHQ-Test"]
    if args.debug:
        splits = ["VFHQ-Test"]

    for split in splits:
        video_names_json = f"../data/VFHQ_datasets_extracted/{split}/extracted_cropped_face_results_25_fps.json"
        video_dir = f"../data/VFHQ_datasets_extracted/{split}/extracted_cropped_face_results_25_fps"
        landmark_output_dir = f"/dev/shm/VFHQ_datasets_extracted/{split}/landmarks_npy_25_fps"

        with open(video_names_json, "r") as f:
            data_dict = json.load(f)
        clips_names = data_dict["clips"]

        current_job_clips = clips_names[args.job_idx :: args.job_num]
        current_gpu_clips = current_job_clips[args.gpu_idx :: args.gpu_num]
        print(
            f"{split} current gpu {args.gpu_idx}/{args.gpu_num} job {args.job_idx}/{args.job_num} needs to process {len(current_gpu_clips)} clips."
        )

        for clip_name in tqdm(
            current_gpu_clips, desc=f"job {args.job_idx}/{args.job_num} gpu {args.gpu_idx}/{args.gpu_num} {split}"
        ):
            frames = data_dict[clip_name]["frames"]
            video_name = data_dict[clip_name]["video_name"]

            for frame_name in frames:
                frame_path = os.path.join(video_dir, video_name, clip_name, frame_name)
                img = cv2.imread(frame_path)
                img = crop_square(img)
                boxes = face_boxes(img)
                n = len(boxes)
                if n == 0:
                    print(f"{frame_path} No face detected, exit")
                    continue

                param_lst, roi_box_lst = tddfa(img, boxes)
                if len(param_lst) == 0:
                    print(f"{frame_path} No face detected, exit")
                    continue
                if len(roi_box_lst) == 0:
                    print(f"{frame_path} No face detected, exit")
                    continue
                ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)
                pts_out_path = os.path.join(
                    landmark_output_dir, video_name, clip_name, frame_name.replace(".png", "_pts.npy")
                )
                pose_out_path = os.path.join(
                    landmark_output_dir, video_name, clip_name, frame_name.replace(".png", "_pose.npy")
                )
                os.makedirs(os.path.dirname(pts_out_path), exist_ok=True)
                one_param = param_lst[0]
                one_ver = ver_lst[0]
                np.save(pts_out_path, one_ver)
                P, pose = calc_pose(one_param)
                np.save(pose_out_path, pose)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--job_idx", type=int, default=0)
    parser.add_argument("--job_num", type=int, default=1)
    parser.add_argument("--gpu_idx", type=int, default=0)
    parser.add_argument("--gpu_num", type=int, default=8)
    parser.add_argument("--config", type=str, default="configs/mb1_120x120.yml")
    parser.add_argument("--debug", action="store_true", default=False)

    args = parser.parse_args()

    print(f"job {args.job_idx} gpu {args.gpu_idx} started")

    main(args)
